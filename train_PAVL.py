import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders.LUAD import (LUAD, CenterCrop, RandomCrop,
                                  RandomRotFlip, ToTensor,
                                  TwoStreamBatchSampler)
from dataloaders.brats2019 import (BraTS2019, CenterCrop, RandomCrop,
                             RandomRotFlip, ToTensor,
                             TwoStreamBatchSampler)
from dataloaders.Pancreas import (Pancreas, CenterCrop, RandomCrop,
                                  RandomRotFlip, ToTensor,
                                  TwoStreamBatchSampler)
from networks.net_factory_3d import net_factory_3d
from utils import losses, metrics, ramps
from val_3D import test_all_case
import clip


class TextPromptEncoder(nn.Module):
    def __init__(self, model_name='ViT-B/32'):
        super(TextPromptEncoder, self).__init__()
        if clip is not None:
            self.model, self.preprocess = clip.load(model_name, device='cuda')

        else:
            raise ImportError("clip not installed. Please install it via 'pip install clip-by-openai'")

    def forward(self, text_prompt):
        text_tokens = clip.tokenize([text_prompt]).cuda()
        text_features = self.model.encode_text(text_tokens)
        return text_features

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/BraTS_data_h5', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str,  default='BraTS2019', help='dataset_name')
parser.add_argument('--exp', type=str,
                    default='BraTS_PAVL', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0',
                    help='gpu id')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=25,
                    help='labeled data')
parser.add_argument('--total_sample', type=int, default=250,
                    help='total samples')
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--uncertainty_th', type=float, default=0.1, help='threshold')
parser.add_argument('--text_prompt', type=str, default='A 3D MRI image of the brain with a tumor lesion', help='Text prompt content')
args = parser.parse_args()
def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def getPrototype(features, mask, class_confidence):
    fts = F.interpolate(features, size=mask.shape[-3:], mode='trilinear')
    mask_new = mask.unsqueeze(1)
    masked_features = torch.mul(fts, mask_new)
    masked_fts = torch.sum(masked_features * class_confidence, dim=(2, 3, 4)) / (
                (mask_new * class_confidence).sum(dim=(2, 3, 4)) + 1e-5)
    return masked_fts

def masked_entropy_loss(p, mask, C=2):
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    y1 = mask * y1
    ent = torch.mean(y1)
    return ent

def paDist(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    delta_s: float = 0.2,
    delta_w: float = 0.8,
    N_p: int = 100,
):
    B, C = features.shape[:2]
    spatial = features.shape[2:]
    n_spatial = features.numel() // (B * C)
    var_map = features.var(dim=1)
    var_flat = var_map.view(B, -1)
    lower_q = torch.quantile(var_flat, delta_s, dim=1)
    upper_q = torch.quantile(var_flat, delta_w, dim=1)
    lower_q = lower_q.view(B, *([1] * len(spatial)))
    upper_q = upper_q.view(B, *([1] * len(spatial)))
    mask_reliable = (var_map >= lower_q) & (var_map <= upper_q)
    prototypes = torch.zeros((B, C), device=features.device, dtype=features.dtype)
    sigma_c = torch.zeros((B, C), device=features.device, dtype=features.dtype)
    dist_map = torch.zeros((B, *spatial), device=features.device, dtype=features.dtype)
    coords = mask_reliable.flatten(start_dim=1).nonzero(as_tuple=False)

    for b in range(B):
        for c in range(C):
            mask_bc = (pseudo_labels[b] == c) & mask_reliable[b]
            if not mask_bc.any():
                continue

            var_vals = var_map[b][mask_bc]
            M = var_vals.numel()
            k = min(M, N_p)
            idx_sorted = torch.argsort(var_vals)[:k]
            pos = mask_bc.nonzero(as_tuple=False)[idx_sorted]
            u_i = features[b][:, pos.t().tolist()]
            sigma_i = var_vals[idx_sorted]
            weights = 1.0 / (sigma_i ** 2 + 1e-6)
            rho_c = (weights.unsqueeze(0) * u_i).sum(dim=1) / weights.sum()
            prototypes[b, c] = rho_c.mean()
            sigma_c[b, c] = sigma_i.mean()

    for b in range(B):
        for c in range(C):
            mask_c = (pseudo_labels[b] == c)
            if not mask_c.any():
                continue

            pos = mask_c.nonzero(as_tuple=False)
            u_all = features[b][:, pos.t().tolist()]
            diff_means = u_all - prototypes[b, c].view(-1, 1)
            mean_dist = (diff_means ** 2).sum(dim=0)
            var_diff = (var_map[b][mask_c] - sigma_c[b, c]) ** 2
            dist_map[b][mask_c] = mean_dist + var_diff

    return dist_map

class URM(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(URM, self).__init__()
        hidden_dim = (in_feature + out_feature) // 2
        self.connectors = nn.Sequential(
            nn.Conv3d(in_feature, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, out_feature, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_feature)
        )
        if in_feature != out_feature:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_feature, out_feature, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_feature)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.connectors(x) + self.shortcut(x))

class statm_loss(nn.Module):
    def __init__(self):
        super(statm_loss, self).__init__()

    def forward(self,x, y):
        x = x.view(x.size(0),x.size(1),-1)
        y = y.view(y.size(0),y.size(1),-1)
        x_mean = x.mean(dim=2)#BC
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean-y_mean).pow(2).mean(1)
        return mean_gap.mean()

class clip_tuning(nn.Module):
    def __init__(
        self,
        in_channels: int,
        text_dim: int = 512,
        hidden_dim: int = None,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_channels // 2, 64)

        self.fc1 = nn.Linear(text_dim, hidden_dim)
        self.act = activation()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2 * in_channels)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:

        text = text_emb.to(features.dtype)

        x = self.fc1(text)
        x = self.act(x)
        x = self.drop(x)
        params = self.fc2(x)
        B, twoC = params.shape
        C = twoC // 2

        gamma, beta = params[:, :C], params[:, C:]
        shape = [B, C] + [1] * (features.dim() - 2)
        gamma = gamma.view(*shape)
        beta  = beta.view(*shape)

        return features * gamma + beta

clip_modulator = clip_tuning(in_channels=2, text_dim=512).cuda()

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    patch_size = args.patch_size
    max_iterations = args.max_iterations
    num_classes = 2
    connector = URM(in_feature=256, out_feature=256).cuda()
    criterion_srd = statm_loss().cuda()

    text_encoder = TextPromptEncoder().cuda()
    text_encoder.eval()
    text_proj = nn.Linear(512, 2).cuda()
    with torch.no_grad():
        text_feature = text_encoder('args.text_prompt')
        text_feature = text_feature.float()
        text_feature = text_proj(text_feature)

    def create_model(ema=False):
        net = net_factory_3d(net_type=args.model,
                             in_chns=1, class_num=num_classes)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # db_train = LUAD(base_dir=train_data_path,
    #                     split='train',
    #                     num=None,
    #                     transform=transforms.Compose([
    #                         RandomRotFlip(),
    #                         RandomCrop(args.patch_size),
    #                         ToTensor(),
    #                     ]))
    db_train = BraTS2019(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))
    # db_train = Pancreas(base_dir=train_data_path,
    #                     split='train',
    #                     transform=transforms.Compose([
    #                         RandomCrop(patch_size),
    #                         ToTensor(),
    #                     ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_sample))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(list(model.parameters()) + list(connector.parameters()),
                          lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            noisy_ema_inputs = unlabeled_volume_batch + noise
            ema_inputs = unlabeled_volume_batch

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_onehot = torch.argmax(outputs_soft, dim=1)

            B, C, D, H, W = outputs_soft.shape
            text_feat_expand = text_feature.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            text_feat_expand = text_feat_expand.expand(B, text_feat_expand.shape[1], D, H, W)
            outputs_soft_clip = clip_tuning(outputs_soft, text_feature)

            with torch.no_grad():
                ema_logits_clean = ema_model(ema_inputs)
                ema_probs_clean = torch.softmax(ema_logits_clean, dim=1)

                student_feature = model.featuremap_center

                noisy_ema_logits = ema_model(noisy_ema_inputs)
                noisy_ema_output = noisy_ema_logits
            T = 10
            mc_prob_list = []
            mc_feat_list = []
            for _ in range(T):
                with torch.no_grad():
                    logits_mc = ema_model(unlabeled_volume_batch)
                    mc_prob_list.append(torch.sigmoid(logits_mc / 2.0))
                    mc_feat_list.append(ema_model.featuremap_center)

            mc_probs = torch.stack(mc_prob_list, dim=0)
            uncert_map = mc_probs.std(dim=0)
            avg_feats = torch.stack(mc_feat_list, dim=0).mean(dim=0)
            HQ_fts = avg_feats

            certainty_mask = (uncert_map < args.uncertainty_th).float()
            rectified_probs = ema_probs_clean * certainty_mask
            rectified_labels = rectified_probs.argmax(dim=1)

            fg_conf = ema_probs_clean[:, 1:2, ...]
            fg_proto = getPrototype(HQ_fts, rectified_labels, fg_conf)
            dist_fg = paDist(HQ_fts, rectified_labels, fg_proto)

            bg_conf = ema_probs_clean[:, 0:1, ...]
            bg_mask = (rectified_labels == 0)
            bg_proto = getPrototype(HQ_fts, bg_mask, bg_conf)
            dist_bg = paDist(HQ_fts, bg_mask, bg_proto)

            sel_bg = (dist_fg > dist_bg).float()
            sel_fg = (dist_fg <= dist_bg).float()
            selection_mask = torch.cat([sel_bg, sel_fg], dim=1)

            unlab_hard = outputs_onehot[args.labeled_bs:]
            unlab_oh = torch.zeros_like(ema_probs_clean) \
                .scatter_(1, unlab_hard.unsqueeze(1), 1.0)

            clean_mask = (selection_mask == unlab_oh).float()
            noisy_mask = 1.0 - clean_mask

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = losses.softmax_mse_loss(outputs[args.labeled_bs:],
                                                       noisy_ema_output)
            consistency_loss = torch.sum(noisy_mask * consistency_dist) / (
                        torch.sum(noisy_mask) + 1e-16) + masked_entropy_loss(outputs_soft[args.labeled_bs:], clean_mask,
                                                                             C=2)

            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs][:])
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            mapped_student_feat = connector(student_feature)
            mapped_student_feat_unlabeled = mapped_student_feat[args.labeled_bs:]
            loss_srd = criterion_srd(mapped_student_feat_unlabeled, HQ_fts)
            loss_sim = 1 - F.cosine_similarity(outputs_soft_clip, text_feat_expand, dim=1).mean()

            supervised_loss = 0.5 * (loss_dice + loss_ce)
            loss = supervised_loss + consistency_weight * consistency_loss + 0.2 * loss_sim + loss_srd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('info/srd_loss', loss_srd, iter_num)
            writer.add_scalar('info/loss_sim', loss_sim, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_srd: %f, loss_sim: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_srd.item(), loss_sim.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 500 and iter_num % 50 == 0:
                model.eval()
                if args.dataset_name =="LUAD":
                    avg_metric = test_all_case(model, args.root_path, test_list="val.list", num_classes=2, patch_size=args.patch_size, stride_xy=18, stride_z=4)
                elif args.dataset_name =="Pancreas_CT":
                    avg_metric = test_all_case(model, args.root_path, test_list="test.list", num_classes=2, patch_size=args.patch_size, stride_xy=16, stride_z=16)
                elif args.dataset_name =="Brats2019":
                    avg_metric = test_all_case(model, args.root_path, test_list="val.list", num_classes=2, patch_size=args.patch_size, stride_xy=64, stride_z=64)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (
                    iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
