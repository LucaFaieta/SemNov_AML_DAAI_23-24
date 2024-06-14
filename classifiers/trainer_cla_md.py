import sys
import os
import warnings
import numpy as np

sys.path.append(os.getcwd())
import os.path as osp
import time
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import *
from utils.dist import *
# noinspection PyUnresolvedReferences
from utils.data_utils import H5_Dataset
from datasets.modelnet import *
from datasets.scanobject import *
from models.classifiers import Classifier
from utils.ood_utils import get_confidence, eval_ood_sncore, iterate_data_odin, \
    iterate_data_energy, iterate_data_gradnorm, iterate_data_react, estimate_react_thres, print_ood_output, \
    get_penultimate_feats, get_network_output, get_point_clouds, eval_id_classification, eval_id_misclassification \
         , eval_ood_misclassification
import wandb
from base_args import add_base_args
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from models.common import convert_model_state, logits_entropy_loss
from models.ARPL_utils import Generator, Discriminator



from gradcam.grad_cam_dgcnn import GradCamDGCNN
from classifiers.common import train_epoch_cla, train_epoch_rsmix_exposure, train_epoch_cs
from models.PointBERT import PointBERTG14,PointBERTL14,PointBERTB32

def get_args():
    parser = argparse.ArgumentParser("OOD on point clouds via contrastive learning")
    parser = add_base_args(parser)

    # experiment specific arguments
    parser.add_argument("--augm_set",
                        type=str, default="rw", help="data augmentation choice", choices=["st", "rw"])
    parser.add_argument("--grad_norm_clip",
                        default=-1, type=float, help="gradient clipping")
    parser.add_argument("--num_points",
                        default=1024, type=int, help="number of points sampled for each object view")
    parser.add_argument("--num_points_test",
                        default=2048, type=int, help="number of points sampled for each SONN object - only for testing")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="md-2-sonn-augmCorr")
    parser.add_argument("--wandb_proj", type=str, default="benchmark-3d-ood-cla")
    parser.add_argument("--loss", type=str, default="CE",
                        choices=["CE", "CE_ls", "cosface", "arcface", "subcenter_arcface", "ARPL", "cosine"],
                        help="Which loss to use for training. CE is default")
    parser.add_argument("--cs", action='store_true', help="Enable confusing samples for ARPL")
    parser.add_argument("--cs_gan_lr", type=float, default=0.0002, help="Confusing samples GAN lr")
    parser.add_argument("--cs_beta", type=float, default=0.1, help="Beta loss weight for CS")
    parser.add_argument("--save_feats", type=str, default=None, help="Path where to save feats of penultimate layer")
    parser.add_argument("--fine_tuning", type = str, default = None, help = "Path for the backbone")
    parser.add_argument("--gradcam", type=bool, default=False, help="Compute gradcam heatmaps for test set")
    # Adopt Corrupted data
    # this flag should be set also during evaluation if testing Synth->Real Corr/LIDAR Augmented models
    parser.add_argument("--corruption",
                        type=str, default=None, help="type of corrupted data (lidar,occlusion,all) - default is None")
    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)
    args.tar1 = "none"
    args.tar2 = "none"

    if args.script_mode == 'eval':
        args.batch_size = 1

    return args



def combined_named_parameters(*param_iterators):
          for param_iterator in param_iterators:
              for name, param in param_iterator:
                  yield name, param

def convert_state_dict(state_dict,string,prefix = False):
    new_state_dict = {}
    for entry in state_dict.keys():
        if entry.startswith(string):
            string_list = entry.split(".")
            if prefix:
                string_list[0] = "enco"
            else:
                string_list = string_list[1:]
            new_key = ".".join(string_list)
            new_state_dict[new_key] = state_dict.get(entry)
    return new_state_dict

### data mgmt ###

def get_list_corr_data(opt, severity=None, split="train"):
    assert split in ['train', 'test']

    if opt.src == "SR1":
        prefix = "modelnet_set1"
    elif opt.src == "SR2":
        prefix = "modelnet_set2"
    else:
        raise ValueError(f"Expected SR source but received: {opt.src} ")

    print(f"get_list_corr_data for {prefix} - split {split}")

    # loads corrupted data
    if severity is None:
        severity = [1, 2, 3, 4]
    if opt.corruption == 'lidar' or opt.corruption == 'occlusion':
        print(f"loading {opt.corruption} data")
        root = osp.join(opt.data_root, "ModelNet40_corrupted", opt.corruption)
        file_names = [f"{root}/{prefix}_{split}_{opt.corruption}_sev" + str(i) + ".h5" for i in severity]
        print(f"corr list files: {file_names}\n")
    elif opt.corruption == 'all':
        print("loading both lidar and occlusion data")
        file_names = []
        root_lidar = osp.join(opt.data_root, "ModelNet40_corrupted", "lidar")
        file_names.extend([f"{root_lidar}/{prefix}_{split}_lidar_sev" + str(i) + ".h5" for i in severity])
        root_occ = osp.join(opt.data_root, "ModelNet40_corrupted", "occlusion")
        file_names.extend([f"{root_occ}/{prefix}_{split}_occlusion_sev" + str(i) + ".h5" for i in severity])
        print(f"corr list files: {file_names}\n")
    else:
        raise ValueError(f"Unknown corruption specified: {opt.corruption}")

    # augmentation mgmt
    if opt.script_mode.startswith("eval"):
        augm_set = None
    else:
        # synth -> real augm
        warnings.warn(f"Using RW augmentation set for corrupted data")
        augm_set = transforms.Compose([
            PointcloudToTensor(),
            AugmScale(),
            AugmRotate(axis=[0.0, 1.0, 0.0]),
            AugmRotatePerturbation(),
            AugmTranslate(),
            AugmJitter()
        ])

    corrupted_datasets = []
    for h5_path in file_names:
        corrupted_datasets.append(H5_Dataset(h5_file=h5_path, num_points=opt.num_points, transforms=augm_set))

    return corrupted_datasets


# for training routine
def get_md_loaders(opt):
    assert opt.src.startswith('SR')
    ws, rank = get_ws(), get_rank()
    drop_last = not str(opt.script_mode).startswith('eval')

    if opt.augm_set == 'st':
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),
            AugmScale(lo=2 / 3, hi=3 / 2),
            AugmTranslate(translate_range=0.2)]
    elif opt.augm_set == 'rw':
        # transformation used for Synthetic->Real-World
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),
            AugmScale(),
            AugmRotate(axis=[0.0, 1.0, 0.0]),
            AugmRotatePerturbation(),
            AugmTranslate(),
            AugmJitter()]
    else:
        raise ValueError(f"Unknown augmentation set: {opt.augm_set}")

    print(f"Train transforms: {set_transforms}")
    train_transforms = transforms.Compose(set_transforms)

    train_data = ModelNet40_OOD(  # sampling performed as dataugm
        data_root=opt.data_root,
        train=True,
        num_points=10000,  # sampling as data augm
        class_choice=opt.src,  # modelnet40 or modelnet10,
        transforms=train_transforms
    )

    print(f"{opt.src} train_data len: {len(train_data)}")

    if opt.corruption is not None:
        # load corrupted datasets
        assert opt.augm_set == 'rw'
        l_corr_data = get_list_corr_data(opt)
        assert isinstance(l_corr_data, list)
        assert isinstance(l_corr_data[0], data.Dataset)
        l_corr_data.append(train_data)
        train_data = torch.utils.data.ConcatDataset(l_corr_data)
        print(f"{opt.src} + corruption {opt.corruption} - train data len: {len(train_data)}")

    test_data = ModelNet40_OOD(
        data_root=opt.data_root,
        train=False,
        num_points=opt.num_points,
        class_choice=opt.src,
        transforms=None)

    train_sampler = DistributedSampler(train_data, num_replicas=ws, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_data, num_replicas=ws, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
        sampler=train_sampler, worker_init_fn=init_np_seed)
    test_loader = DataLoader(
        test_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
        sampler=test_sampler, worker_init_fn=init_np_seed)
    return train_loader, test_loader

### for evaluation routine ###
def get_md_eval_loaders(opt):
    assert opt.script_mode.startswith("eval")
    if not str(opt.src).startswith('SR'):
        raise ValueError(f"Unknown modelnet src: {opt.src}")

    train_data = ModelNet40_OOD(
        data_root=opt.data_root,
        train=True,
        num_points=opt.num_points,
        class_choice=opt.src,
        transforms=None)

    print(f"{opt.src} train data len: {len(train_data)}")

    # append corrupted data to train dataset
    if opt.corruption:
        l_corr_data = get_list_corr_data(opt)  # list of corrupted datasets
        assert isinstance(l_corr_data, list)
        assert isinstance(l_corr_data[0], data.Dataset)
        l_corr_data.append(train_data)  # appending clean data to list corrupted datasets
        train_data = torch.utils.data.ConcatDataset(l_corr_data)  # concat Dataset
        print(f"Cumulative (clean+corrupted) train data len: {len(train_data)}")

    # test data (only clean samples)
    test_data = ModelNet40_OOD(
        data_root=opt.data_root,
        train=False,
        num_points=opt.num_points,
        class_choice=opt.src,
        transforms=None)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                              worker_init_fn=init_np_seed, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                             worker_init_fn=init_np_seed, shuffle=False, drop_last=False)
    return train_loader, test_loader


def get_md_react_val_loader(opt):
    print("Building React validation loader...")
    assert opt.script_mode.startswith("eval")
    if not str(opt.src).startswith('SR'):
        raise ValueError(f"Unknown modelnet src: {opt.src}")

    test_data = ModelNet40_OOD(data_root=opt.data_root, train=False, num_points=opt.num_points,
                               class_choice=opt.src, transforms=None)

    print(f"React Val - {opt.src} data len: {len(test_data)}")

    # append corrupted test data
    if opt.corruption:
        print(f"React Val - adding corrupted synthetic data: {opt.corruption}")
        l_corr_data = get_list_corr_data(opt, split='test')  # list of corrupted test datasets
        assert isinstance(l_corr_data, list)
        assert isinstance(l_corr_data[0], data.Dataset)
        l_corr_data.append(test_data)  # appending clean data to list corrupted datasets
        test_data = torch.utils.data.ConcatDataset(l_corr_data)  # concat Dataset
        print(f"React Val - cumulative (clean+corrupted) data len: {len(test_data)}\n")

    val_data = test_data  # note: modelnet synthetic are not used in synth->real eval
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                            worker_init_fn=init_np_seed, shuffle=False, drop_last=False)
    return val_loader


    



def train(opt, config):
    if torch.cuda.device_count() > 1 and is_dist():
        dist.init_process_group(backend='nccl', init_method='env://')
        device_id, device = opt.local_rank, torch.device(opt.local_rank)
        torch.cuda.set_device(device_id)

    rank, world_size = get_rank(), get_ws()
    assert torch.cuda.is_available(), "no cuda device is available"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    set_random_seed(opt.seed)

    print("*" * 30)
    print(f"{rank}/{world_size} process initialized.\n")
    print(f"{rank}/{world_size} arguments: {opt}. \n")
    print("*" * 30)

    assert opt.config is not None and osp.exists(opt.config)

    # setup loggers
    if rank == 0:
        safe_make_dirs([opt.models_dir, opt.tb_dir, opt.backup_dir])
        project_dir = os.getcwd()
        os.system('cp {} {}/'.format(osp.abspath(__file__), opt.backup_dir))
        os.system('cp -r {} {}/'.format(opt.config, opt.backup_dir))
        os.system('cp -r {} {}/'.format(osp.join(project_dir, "models"), opt.backup_dir))
        os.system('cp -r {} {}/'.format(osp.join(project_dir, "datasets"), opt.backup_dir))
        logger = IOStream(path=osp.join(opt.log_dir, f'log_{int(time.time())}.txt'))
        logger.cprint(f"Arguments: {opt}")
        logger.cprint(f"Config: {config}")
        logger.cprint(f"World size: {world_size}\n")
        wandb.login()
        if opt.wandb_name is None:
            opt.wandb_name = opt.exp_name
        wandb.init(project=opt.wandb_proj, group=opt.wandb_group, name=opt.wandb_name,
                   config={'arguments': vars(opt), 'config': config})
    else:
        logger = None

    assert str(opt.src).startswith("SR"), f"Unknown src choice: {opt.src}"
    train_loader, test_loader = get_md_loaders(opt)
    train_synset = eval(opt.src)
    n_classes = len(set(train_synset.values()))
    if rank == 0:
        logger.cprint(f"{opt.src} train synset: {train_synset}")

    if rank == 0:
        logger.cprint(f"Source: {opt.src}\n"
                      f"Num training classes: {n_classes}")

    # BUILD MODEL


    model = Classifier(args=DotConfig(config['model']), num_classes=n_classes, loss=opt.loss, cs=opt.cs)
    model = model.cuda()
    enco_name = str(config['model']['ENCO_NAME']).lower()
    if enco_name == "gdanet":
        model.apply(weight_init_GDA)
    else:
        model.apply(weights_init_normal)


    if opt.use_sync_bn:
        assert torch.cuda.device_count() > 1 and is_dist(), "cannot use SyncBatchNorm without distributed data parallel"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if rank == 0:
        logger.cprint(f"Model: \n{model}\n")
        logger.cprint(f"param count: \n{count_parameters(model) / 1000000 :.4f} M")
        logger.cprint(f"Loss: {opt.loss}\n")

    if torch.cuda.device_count() > 1 and is_dist():
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    if rank == 0:
        wandb.watch(model, log="gradients")

    # optimizer and scheduler
    if "pointbert" in enco_name:
        ckt = torch.load(opt.fine_tuning, map_location = 'cpu')
        if "openshape-pointbert-vitb32-rgb" in opt.fine_tuning:
            ckt = convert_state_dict(ckt,"pc_encoder.",prefix = True)
        elif "openshape-pointbert-vitg14-rgb" in opt.fine_tuning:
            ckt = convert_state_dict(ckt.get("state_dict"),"module.",prefix = True)
        elif "openshape-pointbert-vitl14-rgb" in opt.fine_tuningame:
            ckt = convert_state_dict(ckt,"pc_encoder.",prefix = True)
        _,une = model.load_state_dict(ckt,strict = False)
        print(f"Unexpercted keys: {une}")
        print(f"Model: \n{model}\n")
        print(f"param count: \n{count_parameters(model) / 1000000 :.4f} M")
        print(f"Loss: {opt.loss}\n")
        parameters = combined_named_parameters(model.head.named_parameters(),model.penultimate.named_parameters())
        optimizer, scheduler = get_opti_sched(parameters, config)
    else:
        optimizer, scheduler = get_opti_sched(model.named_parameters(), config)
    scaler = GradScaler(enabled=opt.use_amp)
    netG, netD = None, None
    optimizerG, optimizerD = None, None
    criterionD = None
    if opt.cs:
        print("Creating GAN for confusing samples")
        netG = Generator(num_points=opt.num_points).cuda()
        netD = Discriminator().cuda()
        criterionD = nn.BCELoss()
        # move to distributed
        if torch.cuda.device_count() > 1 and is_dist():
            netG = DDP(netG, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
            netD = DDP(netD, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
        optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.cs_gan_lr, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.cs_gan_lr, betas=(0.5, 0.999))



    start_epoch = 1
    glob_it = 0
    # TRAINER
    if opt.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ckt = torch.load(opt.resume, map_location=map_location)
        #print(ckt["model"].keys())
        model.load_state_dict(ckt['model'], strict=True)
        if opt.script_mode != 'train_exposure':
            # resume experiment
            optimizer.load_state_dict(ckt['optimizer'])
            scheduler.load_state_dict(ckt['scheduler'])
            if opt.cs:
                netG.load_state_dict(ckt['netG'])
                netD.load_state_dict(ckt['netD'])
            if scaler is not None:
                assert 'scaler' in ckt.keys(), "No scaler key in ckt"
                assert ckt['scaler'] is not None, "None scaler object in ckt"
                scaler.load_state_dict(ckt['scaler'])
            if rank == 0:
                logger.cprint("Restart training from checkpoint %s" % opt.resume)
            start_epoch += int(ckt['epoch'])
            glob_it += (int(ckt['epoch']) * len(train_loader))
        else:
            # load model weights for OE finetuning
            print(f"Finetuning model {opt.resume} for outlier exposure")
        del ckt
    opt.glob_it = glob_it  # will be update by the train_epoch fun.
    opt.gan_glob_it = glob_it
    best_epoch, best_acc = -1, -1
    time1 = time.time()
    for epoch in range(start_epoch, opt.epochs + 1):
        is_best = False
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)

        if opt.script_mode == 'train_exposure':
            # finetuning clf for Outlier Exposure with mixup data
            train_epoch_rsmix_exposure(
                epoch=epoch,
                args=opt,
                train_loader=train_loader,
                model=model,
                scaler=scaler,
                optimizer=optimizer,
                logger=logger)
        else:
            # training clf from scratch
            if opt.cs:
                # train gan for ARPL
                train_epoch_cs(
                    epoch=epoch,
                    args=opt,
                    train_loader=train_loader,
                    model=model,
                    netD=netD,
                    netG=netG,
                    scaler=scaler,
                    optimizer=optimizer,
                    criterionD=criterionD,
                    optimizerD=optimizerD,
                    optimizerG=optimizerG,
                    logger=logger)

            train_epoch_cla(
                epoch=epoch,
                args=opt,
                train_loader=train_loader,
                model=model,
                scaler=scaler,
                optimizer=optimizer,
                logger=logger)

        # step lr
        scheduler.step(epoch)

        # evaluation for classification
        if epoch % opt.eval_step == 0:
            _, src_pred, src_labels = get_confidence(model, test_loader)
            src_pred = to_numpy(src_pred)
            src_labels = to_numpy(src_labels)
            epoch_acc = accuracy_score(src_labels, src_pred)
            epoch_bal_acc = balanced_accuracy_score(src_labels, src_pred)
            if rank == 0:
                logger.cprint(f"Test [{epoch}/{opt.epochs}]\tAcc: {epoch_acc:.4f}, Bal Acc: {epoch_bal_acc:.4f}")
                wandb.log({"test/ep_acc": epoch_acc, "test/ep_bal_acc": epoch_bal_acc, "test/epoch": epoch})
                is_best = epoch_acc >= best_acc
                if is_best:
                    best_acc = epoch_acc
                    best_epoch = epoch

        # save checkpoint
        if rank == 0:
            ckt_path = osp.join(opt.models_dir, "model_last.pth")
            save_checkpoint(opt, ckt_path, model, optimizer, scheduler, scaler, config, epoch)
            if is_best:
                os.system('cp -r {} {}'.format(ckt_path, osp.join(opt.models_dir, f"model_best.pth")))
            if epoch % opt.save_step == 0:
                os.system('cp -r {} {}'.format(ckt_path, osp.join(opt.models_dir, f"model_ep{epoch}.pth")))
    train_time = time.time() - time1
    if rank == 0:
        logger.cprint(f"Training finished - best test acc: {best_acc:.4f} at ep.: {best_epoch}, time: {train_time}")


def eval_ood_md2sonn(opt, config):
    print(f"Arguments: {opt}")
    set_random_seed(opt.seed)

    dataloader_config = {
        'batch_size': opt.batch_size, 'drop_last': False, 'shuffle': False,
        'num_workers': opt.num_workers, 'sampler': None, 'worker_init_fn': init_np_seed}

    # whole evaluation is done on ScanObject RW data
    sonn_args = {
        'data_root': opt.data_root,
        'sonn_split': opt.sonn_split,
        'h5_file': opt.sonn_h5_name,
        'split': 'all',  # we use both training (unused) and test samples during evaluation
        'num_points': opt.num_points_test,  # default: use all 2048 sonn points to avoid sampling randomicity
        'transforms': None  # no augmentation applied at inference time
    }
    
    train_loader, _ = get_md_eval_loaders(opt)
    if opt.src == 'SR1':
        print("Src is SR1\n")
        id_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet1", **sonn_args), **dataloader_config)
        ood1_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet2", **sonn_args), **dataloader_config)
    elif opt.src == 'SR2':
        print("Src is SR2\n")
        id_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet2", **sonn_args), **dataloader_config)
        ood1_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet1", **sonn_args), **dataloader_config)
    else:
        raise ValueError(f"OOD evaluation - wrong src: {opt.src}")
    
    # second SONN out-of-distribution set is common to both SR1 and SR2 sources
    # these are the samples from SONN categories with poor mapping to ModelNet categories
    ood2_loader = DataLoader(ScanObject(class_choice="sonn_ood_common", **sonn_args), **dataloader_config)

    classes_dict = eval(opt.src)
    n_classes = len(set(classes_dict.values()))
    ft = False
    if "openshape-pointbert-vitb32-rgb" in opt.ckpt_path:
            
        #map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ft = True
        ckt = torch.load(opt.ckpt_path, map_location = 'cpu')
        ckt = convert_state_dict(ckt,"pc_encoder.")
        model = PointBERTB32()
        model.load_state_dict(ckt)
        #model.load_state_dict(convert_state_dict2(ckt.get("state_dict")), strict=True)
        print(f"Model: \n{model}\n")
        print(f"param count: \n{count_parameters(model) / 1000000 :.4f} M")
        print(f"Loss: {opt.loss}\n")
    elif "openshape-pointbert-vitg14-rgb" in opt.ckpt_path:
        ckt = torch.load(opt.ckpt_path, map_location = 'cpu')
        ckt = convert_state_dict(ckt.get("state_dict"),"module.")
        model = PointBERTG14()
        model.load_state_dict(ckt)
        #model.load_state_dict(convert_state_dict2(ckt.get("state_dict")), strict=True)
        print(f"Model: \n{model}\n")
        print(f"param count: \n{count_parameters(model) / 1000000 :.4f} M")
        print(f"Loss: {opt.loss}\n")
        ft = True
    elif "openshape-pointbert-vitl14-rgb" in opt.ckpt_path:
        ckt = torch.load(opt.ckpt_path, map_location = 'cpu')
        ckt = convert_state_dict(ckt,"pc_encoder.")
        model = PointBERTL14()
        model.load_state_dict(ckt)
        #model.load_state_dict(convert_state_dict2(ckt.get("state_dict")), strict=True)
        print(f"Model: \n{model}\n")
        print(f"param count: \n{count_parameters(model) / 1000000 :.4f} M")
        print(f"Loss: {opt.loss}\n")
        ft = True

        
    else:
        model = Classifier(args=DotConfig(config['model']), num_classes=n_classes, loss=opt.loss, cs=opt.cs)
        ckt_weights = torch.load(opt.ckpt_path, map_location='cpu')['model']
        ckt_weights = sanitize_model_dict(ckt_weights)
        ckt_weights = convert_model_state(ckt_weights, model.state_dict())
        print(f"Model params count: {count_parameters(model) / 1000000 :.4f} M")
        print("Load weights: ", model.load_state_dict(ckt_weights, strict=True))
    model = model.cuda().eval()
    #print(id_loader.dataset.datas.shape)
    train_feats, train_labels = get_penultimate_feats(model, train_loader,ft)
    src_feats, src_labels = get_penultimate_feats(model, id_loader,ft)
    tar1_feats, tar1_labels = get_penultimate_feats(model, ood1_loader,ft)
    tar2_feats, tar2_labels = get_penultimate_feats(model, ood2_loader,ft)
    
    src_pcds =None
    all_pcds = None
    if not ft:
        if opt.gradcam:
            gradcam = GradCamDGCNN(model, 'positive')
            src_pcds = []
            tar1_pcds = []
            tar2_pcds = []
            #get o3d pcd object for each sample
            src_pcds, tar1_pcds, tar2_pcds = get_point_clouds([id_loader, ood1_loader, ood2_loader])
            all_pcds = src_pcds + tar1_pcds + tar2_pcds
        
        
        print("\n" + "#" * 80)
        print("Computing OOD metrics with distance from train features...")   
        
        src_logits, src_pred, src_labels = get_network_output(model, id_loader)
        tar1_logits, tar1_pred, tar1_labels = get_network_output(model, ood1_loader)
        tar2_logits, tar2_pred, tar2_labels = get_network_output(model, ood2_loader)

        # MSP
        print("\n" + "#" * 80)
        print("Computing OOD metrics with MSP normality score...")
        src_MSP_scores = F.softmax(src_logits, dim=1).max(1)[0]
        tar1_MSP_scores = F.softmax(tar1_logits, dim=1).max(1)[0]
        tar2_MSP_scores = F.softmax(tar2_logits, dim=1).max(1)[0]
        
        if opt.gradcam:
            print("Computing threshold to store ID heatmaps:")
            
            #compute threshold for heatmaps
            scores = torch.tensor(src_MSP_scores)
            scores_np = scores.cpu().numpy()
            thres = np.percentile(scores_np, 0.75 * 100) 
            print(f"  threshold = {thres:.4f}")

            save_path = f"/content/drive/MyDrive/SemNov_AML_DAAI_23-24/gradcam_heatmaps/best_id/MSP_{opt.src}"
            eval_id_classification(gradcam, save_path, src_MSP_scores, src_pred, src_labels, src_pcds, threshold=thres)
            save_path = f"/content/drive/MyDrive/SemNov_AML_DAAI_23-24/gradcam_heatmaps/misclass/MSP_{opt.src}"
            eval_ood_misclassification(gradcam, save_path, [src_MSP_scores, tar1_MSP_scores, tar2_MSP_scores],
                                        [src_pred, tar1_pred, tar2_pred], [src_labels, tar1_labels, tar2_labels],
                                            all_pcds, threshold=0.7)

        eval_ood_sncore(
            scores_list=[src_MSP_scores, tar1_MSP_scores, tar2_MSP_scores],
            preds_list=[src_pred, None, None],  # computes also MSP accuracy on ID test set
            labels_list=[src_labels, None, None],  # computes also MSP accuracy on ID test set
            src_label=1)

        print("#" * 80)


        # MLS
        print("\n" + "#" * 80)
        print("Computing OOD metrics with MLS normality score...")
        src_MLS_scores = src_logits.max(1)[0]
        tar1_MLS_scores = tar1_logits.max(1)[0]
        tar2_MLS_scores = tar2_logits.max(1)[0]
        
        if opt.gradcam:
            gradcam = GradCamDGCNN(model, 'positive')
            #model = model.cuda().eval()
            for name, param in model.enco.conv5.named_parameters():
                print(f"{name} requires_grad: {param.requires_grad}")
            print("Computing threshold to store ID heatmaps:")
            
            #compute threshold for heatmaps
            scores = torch.tensor(src_MLS_scores)
            scores_np = scores.cpu().numpy()
            thres = np.percentile(scores_np, 0.75 * 100) 
            print(f"  threshold = {thres:.4f}")

            save_path = f"/content/drive/MyDrive/SemNov_AML_DAAI_23-24/gradcam_heatmaps/best_id/MLS_{opt.src}"
            eval_id_classification(gradcam, save_path, src_MLS_scores, src_pred, src_labels, src_pcds, threshold=thres)
            save_path = f"/content/drive/MyDrive/SemNov_AML_DAAI_23-24/gradcam_heatmaps/misclass/MLS_{opt.src}"
            eval_ood_misclassification(gradcam, save_path, [src_MLS_scores, tar1_MLS_scores, tar2_MLS_scores],
                                        [src_pred, tar1_pred, tar2_pred], [src_labels, tar1_labels, tar2_labels],
                                            all_pcds, threshold=0.7)    
            
        
        eval_ood_sncore(
            scores_list=[src_MLS_scores, tar1_MLS_scores, tar2_MLS_scores],
            preds_list=[src_pred, None, None],  # computes also MSP accuracy on ID test set
            labels_list=[src_labels, None, None],  # computes also MSP accuracy on ID test set
            src_label=1)
        print("#" * 80)

        # entropy
        print("\n" + "#" * 80)
        src_entropy_scores = 1 / logits_entropy_loss(src_logits)
        tar1_entropy_scores = 1 / logits_entropy_loss(tar1_logits)
        tar2_entropy_scores = 1 / logits_entropy_loss(tar2_logits)
        print("Computing OOD metrics with entropy normality score...")
        eval_ood_sncore(
            scores_list=[src_entropy_scores, tar1_entropy_scores, tar2_entropy_scores],
            preds_list=[src_pred, None, None],  # computes also MSP accuracy on ID test set
            labels_list=[src_labels, None, None],  # computes also MSP accuracy on ID test set
            src_label=1)
        print("#" * 80)
        

        # ODIN
        print("\n" + "#" * 80)
        print("Computing OOD metrics with ODIN normality score...")
        src_odin = iterate_data_odin(model, id_loader)
        tar1_odin = iterate_data_odin(model, ood1_loader)
        tar2_odin = iterate_data_odin(model, ood2_loader)
        eval_ood_sncore(scores_list=[src_odin, tar1_odin, tar2_odin], src_label=1)
        print("#" * 80)

        # Energy
        print("\n" + "#" * 80)
        print("Computing OOD metrics with Energy normality score...")
        src_energy = iterate_data_energy(model, id_loader)
        tar1_energy = iterate_data_energy(model, ood1_loader)
        tar2_energy = iterate_data_energy(model, ood2_loader)
        eval_ood_sncore(scores_list=[src_energy, tar1_energy, tar2_energy], src_label=1)
        print("#" * 80)

        # GradNorm
        print("\n" + "#" * 80)
        print("Computing OOD metrics with GradNorm normality score...")
        src_gradnorm = iterate_data_gradnorm(model, id_loader)
        tar1_gradnorm = iterate_data_gradnorm(model, ood1_loader)
        tar2_gradnorm = iterate_data_gradnorm(model, ood2_loader)
        eval_ood_sncore(scores_list=[src_gradnorm, tar1_gradnorm, tar2_gradnorm], src_label=1)
        print("#" * 80)

        # React with id-dependent threshold
        print("\n" + "#" * 80)
        val_loader = get_md_react_val_loader(opt)
        threshold = estimate_react_thres(model, val_loader)
        print(f"Computing OOD metrics with React (+Energy) normality score, ID-dependent threshold (={threshold:.4f})...")
        print(f"React - using {opt.src} test to compute threshold")
        src_react = iterate_data_react(model, id_loader, threshold=threshold)
        tar1_react = iterate_data_react(model, ood1_loader, threshold=threshold)
        tar2_react = iterate_data_react(model, ood2_loader, threshold=threshold)
        eval_ood_sncore(scores_list=[src_react, tar1_react, tar2_react], src_label=1)
        print("#" * 80)
        
    
     # FEATURES EVALUATION
    all_feats = [train_feats, src_feats, tar1_feats, tar2_feats]
    all_labels = [train_labels, src_labels, tar1_labels, tar2_labels]
    eval_OOD_with_feats(model, opt, all_feats, all_labels, src_pcds, all_pcds ,save_feats=opt.save_feats)


   
    return


def eval_OOD_with_feats(model, opt, all_feats, all_labels,  src_pcds=None, all_pcds=None, save_feats=None):
    from knn_cuda import KNN
    knn = KNN(k=1, transpose_mode=True)

    train_feats, src_feats, tar1_feats, tar2_feats = all_feats[0], all_feats[1], \
                                                      all_feats[2], all_feats[3]
    train_labels, src_labels, tar1_labels, tar2_labels = all_labels[0], all_labels[1], \
                                                          all_labels[2], all_labels[3]


    train_labels = train_labels.cpu().numpy()
    labels_set = set(train_labels)
    prototypes = torch.zeros((len(labels_set), train_feats.shape[1]), device=train_feats.device)
    
    for idx, lbl in enumerate(labels_set):
        mask = train_labels == lbl
        prototype = train_feats[mask].mean(0)
        prototypes[idx] = prototype

    if save_feats is not None:
        if isinstance(train_loader.dataset, ModelNet40_OOD):
            labels_2_names = {v: k for k, v in train_loader.dataset.class_choice.items()}
        else:
            labels_2_names = {}

        output_dict = {}
        output_dict["labels_2_names"] = labels_2_names
        output_dict["train_feats"], output_dict["train_labels"] = train_feats.cpu(), train_labels
        output_dict["id_data_feats"], output_dict["id_data_labels"] = src_feats.cpu(), src_labels
        output_dict["ood1_data_feats"], output_dict["ood1_data_labels"] = tar1_feats.cpu(), tar1_labels
        output_dict["ood2_data_feats"], output_dict["ood2_data_labels"] = tar2_feats.cpu(), tar2_labels
        torch.save(output_dict, save_feats)
        print(f"Features saved to {save_feats}")

    ################################################
    print("Euclidean distances in a non-normalized space:")
    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(train_feats.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = (1 / src_dist).to('cuda')
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample
    src_pred = torch.tensor(src_pred).to('cuda')


    # OOD tar1
    tar1_dist, tar1_ids = knn(train_feats.unsqueeze(0), tar1_feats.unsqueeze(0))
    tar1_dist = tar1_dist.squeeze().cpu()
    tar1_ids = tar1_ids.squeeze().cpu()
    tar1_scores = (1 / tar1_dist).to('cuda')
    tar1_pred = np.asarray([train_labels[i] for i in tar1_ids])
    tar1_pred = torch.tensor(tar1_pred).to('cuda')

    # OOD tar2
    tar2_dist, tar2_ids = knn(train_feats.unsqueeze(0), tar2_feats.unsqueeze(0))
    tar2_dist = tar2_dist.squeeze().cpu()
    tar2_ids = tar2_ids.squeeze().cpu()
    tar2_scores = (1 / tar2_dist).to('cuda')
    tar2_pred = np.asarray([train_labels[i] for i in tar2_ids])
    tar2_pred = torch.tensor(tar2_pred).to('cuda')

    if opt.gradcam:
      gradcam = GradCamDGCNN(model, 'positive')
      print("Computing threshold to store ID heatmaps:")
      
      #compute threshold for heatmaps
      scores = torch.tensor(src_scores)
      scores_np = scores.cpu().numpy()
      thres = np.percentile(scores_np, 0.75 * 100) 
      print(f"  threshold = {thres:.4f}")

      save_path = f"/content/drive/MyDrive/SemNov_AML_DAAI_23-24/gradcam_heatmaps/misclass/euclid_{opt.src}"
      eval_id_classification(gradcam, save_path, src_scores, src_pred, src_labels, src_pcds, threshold=thres)
      eval_ood_misclassification(gradcam, save_path, [src_scores, tar1_scores, tar2_scores],
                                   [src_pred, tar1_pred, tar2_pred], [src_labels, tar1_labels, tar2_labels],
                                      all_pcds, threshold=0.7)     


    eval_ood_sncore(
        scores_list=[src_scores, tar1_scores, tar2_scores],
        preds_list=[src_pred, None, None],  # [src_pred, None, None],
        labels_list=[src_labels, None, None],  # [src_labels, None, None],
        src_label=1  # confidence should be higher for ID samples
    )

    print("\nEuclidean distances with prototypes:")
    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(prototypes.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = 1 / src_dist
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    # OOD tar1
    tar1_dist, _ = knn(prototypes.unsqueeze(0), tar1_feats.unsqueeze(0))
    tar1_dist = tar1_dist.squeeze().cpu()
    tar1_scores = 1 / tar1_dist

    # OOD tar2
    tar2_dist, _ = knn(prototypes.unsqueeze(0), tar2_feats.unsqueeze(0))
    tar2_dist = tar2_dist.squeeze().cpu()
    tar2_scores = 1 / tar2_dist

    eval_ood_sncore(
        scores_list=[src_scores, tar1_scores, tar2_scores],
        preds_list=[src_pred, None, None],
        labels_list=[src_labels, None, None],
        src_label=1  # confidence should be higher for ID samples
    )

    ################################################
    print("\nCosine similarities on the hypersphere:")
    # cosine sim in a normalized space
    train_feats = F.normalize(train_feats, p=2, dim=1)
    src_feats = F.normalize(src_feats, p=2, dim=1)
    tar1_feats = F.normalize(tar1_feats, p=2, dim=1)
    tar2_feats = F.normalize(tar2_feats, p=2, dim=1)
    src_scores, src_ids = torch.mm(src_feats, train_feats.t()).max(1)
    tar1_scores, tar1_ids = torch.mm(tar1_feats, train_feats.t()).max(1)
    tar2_scores, tar2_ids = torch.mm(tar2_feats, train_feats.t()).max(1)
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample
    tar1_pred = np.asarray([train_labels[i] for i in tar1_ids])
    tar2_pred = np.asarray([train_labels[i] for i in tar2_ids])
    src_pred = torch.tensor(src_pred).to('cuda')
    tar1_pred = torch.tensor(tar1_pred).to('cuda')
    tar2_pred = torch.tensor(tar2_pred).to('cuda')

    if opt.gradcam:
      gradcam = GradCamDGCNN(model, 'positive')
      print("Computing threshold to store ID heatmaps:")
      
      #compute threshold for heatmaps
      scores = torch.tensor(src_scores)
      scores_np = scores.cpu().numpy()
      thres = np.percentile(scores_np, 0.75 * 100) 
      print(f"  threshold = {thres:.4f}")


      save_path = f"/content/drive/MyDrive/SemNov_AML_DAAI_23-24/gradcam_heatmaps/misclass/cosine{opt.src}"
      eval_id_classification(gradcam, save_path, src_scores, src_pred, src_labels, src_pcds, threshold=thres)
      eval_ood_misclassification(gradcam, save_path, [(0.5 * src_scores + 0.5), (0.5 * tar1_scores + 0.5), (0.5 * tar2_scores + 0.5)],
                                   [src_pred, tar1_pred, tar2_pred], [src_labels, tar1_labels, tar2_labels],
                                      all_pcds, threshold=0.7)   



    eval_ood_sncore(
        scores_list=[(0.5 * src_scores + 0.5).cpu(), (0.5 * tar1_scores + 0.5).cpu(), (0.5 * tar2_scores + 0.5).cpu()],
        preds_list=[src_pred, None, None],  # [src_pred, None, None],
        labels_list=[src_labels, None, None],  # [src_labels, None, None],
        src_label=1  # confidence should be higher for ID samples
    )
    print("\nCosine similarities with prototypes:")
    # cosine sim in a normalized space
    prototypes = F.normalize(prototypes, p=2, dim=1)
    src_feats = F.normalize(src_feats, p=2, dim=1)
    tar1_feats = F.normalize(tar1_feats, p=2, dim=1)
    tar2_feats = F.normalize(tar2_feats, p=2, dim=1)
    src_scores, src_ids = torch.mm(src_feats, prototypes.t()).max(1)
    tar1_scores, _ = torch.mm(tar1_feats, prototypes.t()).max(1)
    tar2_scores, _ = torch.mm(tar2_feats, prototypes.t()).max(1)
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample
    eval_ood_sncore(
        scores_list=[(0.5 * src_scores + 0.5).cpu(), (0.5 * tar1_scores + 0.5).cpu(), (0.5 * tar2_scores + 0.5).cpu()],
        preds_list=[src_pred, None, None],
        labels_list=[src_labels, None, None],
        src_label=1  # confidence should be higher for ID samples
    )
    print("#" * 80)


def main():
    args = get_args()
    config = load_yaml(args.config)

    if args.script_mode.startswith('train'):
        # launch trainer
        print("training...")
        assert args.checkpoints_dir is not None and len(args.checkpoints_dir)
        assert args.exp_name is not None and len(args.exp_name)
        args.log_dir = osp.join(args.checkpoints_dir, args.exp_name)
        args.tb_dir = osp.join(args.checkpoints_dir, args.exp_name, "tb-logs")
        args.models_dir = osp.join(args.checkpoints_dir, args.exp_name, "models")
        args.backup_dir = osp.join(args.checkpoints_dir, args.exp_name, "backup-code")
        train(args, config)
    else:
        # eval Modelnet -> SONN
        assert args.ckpt_path is not None and len(args.ckpt_path)
        print("out-of-distribution eval - Modelnet -> SONN ..")
        eval_ood_md2sonn(args, config)


if __name__ == '__main__':
    main()
