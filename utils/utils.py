import os
import argparse
import numpy as np
import yaml
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
import cv2
import open3d as o3d
from torch.autograd import Variable

try:
    from torchlars import LARS
except ImportError:
    LARS = None
    print("Cannot import torchlars")


def save_checkpoint(opt, ckt_path, model, optimizer, scheduler, scaler, config, epoch, netG=None, netD=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'args': opt,
        'config': config,
        'epoch': epoch
    }
    if netG is not None:
        state["netG"] = netG.state_dict()
    if netD is not None:
        state["netD"] = netD.state_dict()
    torch.save(state, ckt_path)
    del state


def count_parameters(model):
    """Count number of trainable parameters in a model 

    Args:
        model (nn.Module): model for which to count params

    Returns 
        total_params (int): params count
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    return total_params


class DotConfig:
    """
    Access to dictionary through dot notation - more troubles than benefits
    """

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v


class IOStream:
    """
    pretty logger
    """

    def __init__(self, path):
        self.f = open(path, 'a')
        self.blue = lambda x: '\033[94m' + x + '\033[0m'
        self.red = lambda x: '\033[31m' + x + '\033[0m'

    def cprint(self, text, color=None):
        if color is not None and (color == 'b' or color == 'blue'):
            print(self.blue(text))
        elif color is not None and (color == 'r' or color == 'red'):
            print(self.red(text))
        else:
            print(text)

        self.f.write(text + '\n')
        self.f.flush()

    def fprint(self, text):
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def print_ok(pstr):
    print('\033[92m' + pstr + '\033[0m')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def sanitize_model_dict(state_dict, to_remove_str='module'):
    """
    Args:
        state_dict: dict to update removing prefix from keys
        to_remove_str: prefix to remove from keys

    Returns:
        new_state_dict: updated state dict
    """
    new_state_dict = OrderedDict()
    remove_len = len(to_remove_str) + 1
    for k, v in state_dict.items():
        if str(k).startswith(to_remove_str):
            name = k[remove_len:]  # remove to_remove_str
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_yaml(config_path):
    assert os.path.exists(config_path), f"wrong config path: {config_path}"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


# # https://github.com/rwightman/pytorch-image-models/blob/c5a8e929fb746dc1ff85bee980b41ce3eb24f600/timm/optim/optim_factory.py#L35
# def param_groups_weight_decay(
#         model: nn.Module,
#         weight_decay=1e-5,
#         no_weight_decay_list=()
# ):
#     """ by default weight decay is not applied on bias parameters """
#     no_weight_decay_list = set(no_weight_decay_list)
#     decay = []
#     no_decay = []
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#
#         if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
#             no_decay.append(param)
#         else:
#             decay.append(param)
#
#     return [
#         {'params': no_decay, 'weight_decay': 0.},
#         {'params': decay, 'weight_decay': weight_decay}]


# @torch.jit.ignore
# def no_weight_decay(self):
#     nwd = {'pos_embed', 'cls_token'}
#     for n, _ in self.named_parameters():
#         if 'relative_position_bias_table' in n:
#             nwd.add(n)
#     return nwd


def param_groups_weight_decay(named_params, weight_decay, no_weight_decay_list=()):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []

    for curr_name, curr_param in named_params:
        if not curr_param.requires_grad:
            continue
        elif any(layer_name in curr_name for layer_name in no_weight_decay_list):
            print(f'Param: {curr_name} excluded from weight_decay')
            no_decay.append(curr_param)
        else:
            decay.append(curr_param)

    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]


def get_opti_sched(named_params, config):
    # optimizer
    opt_config = config['optimizer']
    weight_decay = opt_config["weight_decay"]  # weight decay to apply
    skip_wd_list = opt_config["skip_wd"]  # layers excluded from wd - e.g. ['bias', 'cls_token']
    assert isinstance(skip_wd_list, list)
    parameters = param_groups_weight_decay(named_params, weight_decay, skip_wd_list)

    # get optimizer
    opt_name = str(opt_config["type"]).lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            params=parameters,
            **opt_config['kwargs']
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            params=parameters,
            **opt_config['kwargs']
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params=parameters,
            **opt_config['kwargs']
        )
    elif opt_name == "lars":
        base_optimizer = torch.optim.SGD(
            params=parameters,
            **opt_config['kwargs']
        )
        optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_name}")

    # scheduler
    sche_config = config['scheduler']
    sche_name = str(sche_config["type"]).lower()
    if sche_name == 'coslr':
        scheduler = CosineLRScheduler(optimizer, **sche_config['kwargs'])
    elif sche_name == 'steplr':
        # scheduler = StepLRScheduler(optimizer, **sche_config['kwargs'])
        base_lr = sche_config['kwargs']['base_lr']
        lr_decay = sche_config['kwargs']['lr_decay']
        decay_step = sche_config['kwargs']['decay_step']
        lr_clip = sche_config['kwargs']['lr_clip']
        scheduler = LambdaLR(optimizer, lambda e: max(lr_decay ** (e // decay_step), lr_clip / base_lr))
    else:
        raise ValueError(f"Unknown scheduler type: {sche_name}")

    return optimizer, scheduler


def safe_make_dirs(dirs):
    for currdir in dirs:
        if not os.path.exists(currdir):
            os.makedirs(currdir)


def weights_init_normal(m):
    std = 0.02
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.LayerNorm):
        # https://github.com/lulutang0608/Point-BERT/blob/410ce3ede3991f78b854c8f75aedf9c275aa3529/models/Point_BERT.py#L293
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0)


# weight initialization GDA
def weight_init_GDA(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def to_numpy(t):
    if torch.is_tensor(t):
        return t.data.cpu().numpy()
    elif type(t).__module__ == 'numpy':
        return t
    elif isinstance(t, list):
        return np.asarray(t)
    else:
        raise ValueError(f"t is {type(t)}")


def gather_by_idxs(source, idx):
    """

    :param source: input points data, [B, N, C]
    :param idx: sample index data, [B, S]
    :return: indexed points data, [B, S, C]
    """
    B = source.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(source.device).view(view_shape).repeat(repeat_shape)
    new_points = source[batch_indices, idx, :]
    return new_points


def cal_ce_loss(pred, target, smoothing=False):
    """
    Calculate cross entropy loss, apply label smoothing if needed.
    Src: DGCNN pytorch implementation
    """

    target = target.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, target)

    return loss

def pcd_to_o3d(pcd):
    pcd = pcd.squeeze(0).cpu().numpy()
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
    
    return o3d_pcd


def store_heatmap(grad_cam, points, target, save_path):
    points, target = Variable(points), Variable(target)
    input, target = points.cuda(), target.cuda()
    
    heatmaps = []
    points = points[:, :3, :]  # get rid of normals      
    

    #print(f"Input cloud shape: {input.shape}")
    mask = grad_cam(input, target)

    heatmap_input = input.transpose(2,1)
    colored_re_pointcloud, _ = heatmap_plt(heatmap_input, mask)

    heatmaps.append(colored_re_pointcloud)

    pcd = heatmaps[0]

    o3d.io.write_point_cloud(save_path, pcd)

def get_point_clouds(loaders, glb_idx=False):
    src_pcds = []
    tar1_pcds = []
    tar2_pcds = []
    all_pcds = []

    for nload, loader in enumerate(loaders):
        for idx, (data, label) in enumerate(loader):
            #data = pcd_to_o3d(data)
            if not glb_idx:
                if nload == 0:
                    src_pcds.append(data)
                elif nload == 1:
                    tar1_pcds.append(data)
                else:
                    tar2_pcds.append(data)
            else:
                all_pcds.append(data)

    if glb_idx:
        return all_pcds
    else:
        return src_pcds, tar1_pcds, tar2_pcds



def heatmap_plt(input, mask, offset=0, colormap=True):
    if colormap is True:
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    else:
        heatmap = mask
    heatmap = np.float32(heatmap) / 255
    try:
        heatmap = np.squeeze(heatmap, axis=1)
        # heatmap = np.squeeze(heatmap, axis=1)
        heatmap = np.squeeze(heatmap)  # , axis=1
        heatmap[:, [0, 1, 2]] = heatmap[:, [2, 1, 0]]
    except:
        pass
    #print(heatmap.shape)
    pcd_list = []
    pcd_list_orig = []
    for i in range(int(input.size(2))):
        pcd_ = o3d.geometry.PointCloud()
        pcd_list.append(pcd_)
        pcd_list_orig.append(pcd_)

    prc_r_all = input[0].transpose(1, 0).contiguous().data.cpu()

    colored_re_pointcloud2 = o3d.geometry.PointCloud()
    colored_re_pointcloud2_orig = o3d.geometry.PointCloud()

    for j in range(int(input.size(2))):
        current_patch = prc_r_all[j,] + offset
        current_patch = current_patch.unsqueeze(dim=0)
        pcd_list[j].points = o3d.utility.Vector3dVector(current_patch)
        #print(heatmap[j, 0], heatmap[j, 1], heatmap[j, 2])
        pcd_list[j].paint_uniform_color([heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]])
        colored_re_pointcloud2 += pcd_list[j]

        current_patch_orig = prc_r_all[j,]
        current_patch_orig = current_patch_orig.unsqueeze(dim=0)
        pcd_list_orig[j].points = o3d.utility.Vector3dVector(current_patch_orig)
        pcd_list_orig[j].paint_uniform_color([heatmap[j, 0], heatmap[j, 1], heatmap[j, 2]])
        colored_re_pointcloud2_orig += pcd_list_orig[j]

    return colored_re_pointcloud2, colored_re_pointcloud2_orig


def drop_highest_points(input, cam, num_drops=1):
    # copy input
    points = input.clone().detach()
    hcam = np.copy(cam)
    # hcam_gate is necessary to ignore the cam values of the already ignored points
    hcam_add_gate = np.zeros(hcam.shape, dtype=np.dtype(np.float32))
    k = 1

    for i in range(num_drops):
        idx = np.argmax(hcam, axis=0)
        # set cam value to zero so that the next argmax call the return the next lower value
        #np.concatenate((range(hcam.shape[0]), idx), axis=0)
        #idx_indices = np.dstack((np.range(hcam.shape[0]), idx)).squeeze(axis=0)
        for idx_j in range(len(idx)):
            hcam[idx_j, idx[idx_j]] = -1.5
            hcam_add_gate[idx_j, idx[idx_j]] = -2.5
            points[idx_j, :, idx[idx_j]] = 0.0  # shift point to center

        result = None
    print(hcam.shape)
    return points, hcam