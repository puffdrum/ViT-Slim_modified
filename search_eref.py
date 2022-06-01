import argparse
import datetime
from turtle import update
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer, create_optimizer_v2
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, train_one_epoch_partialbp, evaluate
from losses import DistillationLoss, SearchingDistillationLoss, SearchingDistillationLossChannelWise, LossRegZero, LossRegOne
from samplers import RASampler
import models
import utils
from utils import simulated_annealing_sparse_layerwise, update_bp_methods, generate_criterion_w, update_grad_masks
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='weight decay (default: 1e-3)')
    # Learning rate schedule parameters (if sched is none, warmup and min dont matter)
    parser.add_argument('--sched', default='none', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "none"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Dataset parameters
    parser.add_argument('--data-path', default='../imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'IMNET100'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # searching parameters
    parser.add_argument('--w1', default=0.0001, type=float, help='weightage to attn sparsity')
    parser.add_argument('--w2', default=0.0001, type=float, help='weightage to mlp sparsity')
    parser.add_argument('--w3', default=0.0001, type=float, help='weightage to patch sparsity')
    parser.add_argument('--pretrained_path', default='exps/deit_small/checkpoint.pth', type=str)
    parser.add_argument('--head_search', action='store_true')
    parser.add_argument('--uniform_search', action='store_true')
    parser.add_argument('--freeze_weights', action='store_true')
    parser.add_argument('--pre_epochs', default=10, type=int, help='epochs for pre sparse')
    parser.add_argument('--fs_epochs', default=20, type=int, help='epochs for final sparse')
    parser.add_argument('--ms_times', default=3, type=int, help='times for mix and sparse')
    parser.add_argument('--m_epochs', default=10, type=int, help='epochs for mix')
    parser.add_argument('--s_epochs', default=10, type=int, help='epochs for sparse')
    parser.add_argument('--ft', default=0.1, type=float, help='threshold for forgetting')
    parser.add_argument('--rt', default=0.5, type=float, help='threshold for remembering')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    args.epochs=args.pre_epochs+args.ms_times*2*(args.m_epochs+args.s_epochs)+args.fs_epochs
    print(args)
    args.w1/=args.world_size
    args.w2/=args.world_size
    args.w3/=args.world_size

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False, )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        method='search',
        head_search=args.head_search,
        uniform_search=args.uniform_search,
    )
    model.load_state_dict(torch.load(args.pretrained_path)['model'], strict=False)
    model.to(device)
    model.correct_require_grad(args.w1, args.w2, args.w3)

    if args.freeze_weights:
        for name, p in model.named_parameters():
            if "zeta" in name or "norm" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model_without_ddp)
    optimizer0 = create_optimizer(args, model_without_ddp)
    optimizer1 = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy() 

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    #criterion = SearchingDistillationLoss(
    #    criterion, device, attn_w=args.w1, mlp_w=args.w2, patch_w=args.w3
    #)
    #criterion = SearchingDistillationLossLayerWise(
    #    criterion, device, attn_w=args.w1, mlp_w=args.w2, patch_w=args.w3
    #)

    output_dir = Path(args.output_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_soft_accuracy = 0.0
    
    zetas_attn, zetas_mlp, zetas_patch = model.module.give_zetas() # [[], [], []]
    # coefficients for L1-regularization loss
    w_attn = [args.w1 for n in range(len(zetas_attn))]
    w_mlp = [args.w2 for n in range(len(zetas_mlp))]
    w_patch = [args.w3 for n in range(len(zetas_patch))]
    w = []
    w.extend(w_attn)
    w.extend(w_mlp)
    w.extend(w_patch)
    init_w = [args.w1, args.w2, args.w3]
    
    zetas_attn, zetas_mlp, zetas_patch = model.module.give_zetas_layerwise() # [[[],[],...], [[],[],...], [[],[],...]]
    # grad_masks for partial bp of zetas
    grad_masks_attn = [[1 for x in range(len(n))] for n in zetas_attn]
    grad_masks_mlp = [[1 for x in range(len(n))] for n in zetas_mlp]
    grad_masks_patch = [[1 for x in range(len(n))] for n in zetas_patch]
    grad_masks = [grad_masks_attn, grad_masks_mlp, grad_masks_patch]

    # bp_methods for zetas: -1: w*||z||; 1: w*||z-1||; 0: w*||z||+L(accuracy)
    bp_methods_attn = [[0 for x in range(len(n))] for n in zetas_attn]
    bp_methods_mlp = [[0 for x in range(len(n))] for n in zetas_mlp]
    bp_methods_patch = [[0 for x in range(len(n))] for n in zetas_patch]
    bp_methods = [bp_methods_attn, bp_methods_mlp, bp_methods_patch]

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if epoch<args.pre_epochs:
            # pre sparse stage, finished
            criterion_train = SearchingDistillationLossChannelWise(
                criterion, device, w=w
            )
            train_stats = train_one_epoch(
                model, criterion_train, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn, use_amp=False
            )
        elif epoch>=args.epochs-args.fs_epochs:
            # final sparse stage, finished
            if epoch == args.epochs-args.fs_epochs:
                # update bp_methods
                zetas_attn, zetas_mlp, zetas_patch = model.module.give_zetas_layerwise() # [[[],[],...], [[],[],...], [[],[],...]]
                bp_methods = update_bp_methods(zetas_attn, zetas_mlp, zetas_patch, bp_methods, args.ft, args.ft)
                # generate coefficients for losses
                w_accuracy = generate_criterion_w(bp_methods, init_w, 0)
                w_regzero = generate_criterion_w(bp_methods, init_w, -1)
                w_regone = generate_criterion_w(bp_methods, init_w, 1)
                # generate masks for partial regularization bp
                grad_masks = update_grad_masks(grad_masks, bp_methods)
                # generate losses
                criterion_accuracy = SearchingDistillationLossChannelWise(criterion, device, w=w_accuracy)
                criterion_regzero = LossRegZero(device, w=w_regzero)
                criterion_regone = LossRegOne(device, w=w_regone)
            # train
            train_stats = train_one_epoch_partialbp(
                model, criterion_accuracy, criterion_regzero, 
                criterion_regone, grad_masks, data_loader_train,
                optimizer, optimizer0, optimizer1, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn, use_amp=False
            )
        elif ((epoch-args.pre_epochs)//10)%2 == 0:
            # middle mix stage
            if (epoch-args.pre_epochs)%(args.m_epochs+args.s_epochs) == 0:
                # update bp_methods
                zetas_attn, zetas_mlp, zetas_patch = model.module.give_zetas_layerwise() # [[[],[],...], [[],[],...], [[],[],...]]
                bp_methods = update_bp_methods(zetas_attn, zetas_mlp, zetas_patch, bp_methods, args.ft, args.rt)
                # generate coefficients for losses
                w_accuracy = generate_criterion_w(bp_methods, init_w, 0)
                w_regzero = generate_criterion_w(bp_methods, init_w, -1)
                w_regone = generate_criterion_w(bp_methods, init_w, 1)
                # generate masks for partial regularization bp
                grad_masks = update_grad_masks(grad_masks, bp_methods)
                # generate losses
                criterion_accuracy = criterion
                criterion_regzero = LossRegZero(device, w=w_regzero)
                criterion_regone = LossRegOne(device, w=w_regone)
            # train
            train_stats = train_one_epoch_partialbp(
                model, criterion_accuracy, criterion_regzero, 
                criterion_regone, grad_masks, data_loader_train,
                optimizer, optimizer0, optimizer1, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn, use_amp=False
            )
        else:
            # middle sparse stage
            # train
            train_stats = train_one_epoch_partialbp(
                model, criterion_accuracy, criterion_regzero, 
                criterion_regone, grad_masks, data_loader_train,
                optimizer, optimizer0, optimizer1, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn, use_amp=False
            )
        
        if args.output_dir:
            checkpoint_paths = [output_dir / 'running_ckpt.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device, use_amp=False)
        print(f"Soft Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_soft_accuracy = max(max_soft_accuracy, test_stats["acc1"])
        print(f'Max soft accuracy: {max_soft_accuracy:.2f}%')
        
#         if args.output_dir and test_stats["acc1"]>=max_soft_accuracy:
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
#                     'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
                
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'soft_test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        zetas = model.module.give_zetas()
        a=plt.hist(zetas, bins=1000)
        path = output_dir / f'zetas_{epoch}.png'
        plt.savefig(path)
        plt.clf()

        #zetas_attn, zetas_mlp, zetas_patch = model.module.give_zetas_layerwise()
        #cur_zetas = [zetas_attn, zetas_mlp, zetas_patch]
        #if epoch>5:
        #    zetas_search = simulated_annealing_sparse_layerwise(pre_zetas, cur_zetas, epoch, args.epochs)
        #    model.module.update_zetas_SA(zetas_search)

        #pre_zetas = cur_zetas

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT searching script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)