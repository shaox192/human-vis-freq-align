
"""
Train the model with bandpass/blur/none layer on imagenet classification task
train on the subset of fine-grained categories that belong to the 16 basic-level category
"""


import argparse
import os
import random
import time
import warnings
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Subset

import models

# from utils import print_safe, Summary, AverageMeter, ProgressMeter, accuracy, pickle_dump, make_directory, get_rank
import utils
import data_loader

model_names = ["resnet18", ]

parser = argparse.ArgumentParser(description='PyTorch imagenet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--save_dir', required=True, type=str, help='path to save the checkpoints')
parser.add_argument('--img_folder_txt', default="./data/human16-209.txt",
                    type=str, help='path to a textfile of sub categories of imagenet to be used')

parser.add_argument('--category-209', action='store_true', default=True,
                    help='use the 209 fine-grained categories belonged the 16 basic-level categories')
parser.add_argument('--category-16', action='store_true', help='use the 16 basic-level categories')
parser.add_argument('--category-1k', action='store_true', help='use the original 1k categories')

# parser.add_argument('--orig-imagenet-lbs', type=str, help='path to the orig imagenet mapping')

########### model parameters
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument("--append-layer", default="None", type=str, 
                    help="append which layer: [default(None), bandpass, blur] to the beginning of the model")
parser.add_argument("--kernel-size", default=31, type=int, 
                    help="kernel size for the bandpass/blur layer")

############ training parameters
parser.add_argument('--train_workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency per epoch (default: 5)')
parser.add_argument("--save-interval", default=2, type=int,
                    help='checkpointing interval')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='ONLY evaluate model on validation set to get inference performance')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--saved-data', action='store_true', help="use saved data")

#### scheduler
parser.add_argument('--scheduler-gamma', type=float, default=0.1, help='stepLR scheduler gamma')
parser.add_argument('--scheduler-step-size', type=int, default=20, help='stepLR scheduler reduce step size')



def main():
    args = parser.parse_args()

    if args.category_16 or args.category_1k:
        args.category_209 = False

    utils.print_safe("\n***check params ---------")
    for arg in vars(args):
        utils.print_safe(f"{arg}: {getattr(args, arg)}")
    utils.print_safe("--------------------------\n")

    utils.print_safe(f"* [EMPH] learning rate is {args.lr}")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('SEEDING TRAINING: '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    ## -- set up distributed training
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    ## -- set up save directory
    datetime_str = (
      str(datetime.datetime.now().replace(microsecond=0))
      .replace(" ", "-")
      .replace(":", "-")
    )
    category_str = "209" if args.category_209 else "16" if args.category_16 else "1k"
    save_f_name = f"{args.save_dir}/{args.arch}-layer-{args.append_layer}-category-{category_str}-{datetime_str}"

    utils.print_safe(f"*** Saving to: {save_f_name}")
    utils.make_directory(args.save_dir)
    
    ## -- find device
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ## -- load data
    train_loader, val_loader, \
        train_sampler, val_sampler = data_loader.build_data_loader(args)
    
    for i, (images, target) in enumerate(train_loader):
        print(i, images.shape, target)

    utils.print_safe(f"Data loaded: train: {len(train_loader)}, val: {len(val_loader)}", flush=True)

    ## -- create model
    if args.category_16:
        num_classes = 16
    elif args.category_1k:
        num_classes = 1000
    else:
        num_classes = 209
    classifier = models.get_classifier(args.arch, num_classes=num_classes)

    if args.append_layer == "bandpass":
        model = models.BandPassNet(classifier, kernel_size=args.kernel_size)
    elif args.append_layer == "blur":
        raise NotImplementedError
        # model = models.BlurNet(classifier)
    else:
        model = classifier
    utils.print_safe(model)
    exit()

    if not torch.cuda.is_available():
        utils.print_safe('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                print_safe(f"distributing batch size: {args.batch_size}")
                # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    ## -- loss, optimizer, scheduler
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = ManLoss(man_stats, args.decorr_ON).to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                              weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=100)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_safe("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print_safe("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print_safe("=> no checkpoint found at '{}'".format(args.resume))


    if args.evaluate:
        validate(val_loader, model, criterion, device, args, man_stats["man_info_retriev_idx"])
        return

    train_loss_cls_epk = []
    train_loss_reg_orig_rad_epk = []
    train_loss_reg_orig_dim_epk = []
    train_loss_reg_decorr_rad_epk = []
    train_loss_reg_decorr_dim_epk = []
    train_acc1_epk = []
    train_acc5_epk = []

    val_acc1_epk = []
    val_acc5_epk = []
    val_loss_cls_epk = []
    val_loss_reg_orig_rad_epk = []
    val_loss_reg_orig_dim_epk = []
    val_loss_reg_decorr_rad_epk = []
    val_loss_reg_decorr_dim_epk = []

    for epoch in range(args.start_epoch, args.epochs):
        print_safe(f"Epoch: {epoch}, lr: {scheduler.get_last_lr()}, {optimizer.param_groups[0]['lr']}")
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss_classify, train_loss_reg_orig_rad, train_loss_reg_orig_dim, \
        train_loss_reg_decorr_rad, train_loss_reg_decorr_dim, \
        train_top1, train_top5 = train(train_loader, model, criterion, optimizer, \
                                           epoch, device, args, man_stats["man_info_retriev_idx"])

        # evaluate on validation set
        acc1, acc5, loss_classify, val_loss_reg_orig_rad, val_loss_reg_orig_dim, \
            val_loss_reg_decorr_rad, val_loss_reg_decorr_dim = validate(
            val_loader, model, criterion, device, args, man_stats["man_info_retriev_idx"]
        )

        train_loss_cls_epk.append(train_loss_classify)
        train_loss_reg_orig_rad_epk.append(train_loss_reg_orig_rad)
        train_loss_reg_orig_dim_epk.append(train_loss_reg_orig_dim)

        train_loss_reg_decorr_rad_epk.append(train_loss_reg_decorr_rad)
        train_loss_reg_decorr_dim_epk.append(train_loss_reg_decorr_dim)
        train_acc1_epk.append(train_top1)
        train_acc5_epk.append(train_top5)

        val_acc1_epk.append(acc1)
        val_acc5_epk.append(acc5)
        val_loss_cls_epk.append(loss_classify)
        val_loss_reg_orig_rad_epk.append(val_loss_reg_orig_rad)
        val_loss_reg_orig_dim_epk.append(val_loss_reg_orig_dim)

        val_loss_reg_decorr_rad_epk.append(val_loss_reg_decorr_rad)
        val_loss_reg_decorr_dim_epk.append(val_loss_reg_decorr_dim)

        scheduler.step()

        if (
                epoch % args.save_interval == 0 and
                epoch != 0 and
                args.multiprocessing_distributed and
                args.rank % ngpus_per_node == 0
        ):
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict() if args.multiprocessing_distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, f"{save_dir_name}_epk_{epoch}.pth")

            pickle_dump({
                "train_loss_cls_epk": train_loss_cls_epk,
                "train_loss_reg_orig_rad_epk": train_loss_reg_orig_rad_epk,
                "train_loss_reg_orig_dim_epk": train_loss_reg_orig_dim_epk,
                "train_loss_reg_decorr_rad_epk": train_loss_reg_decorr_rad_epk,
                "train_loss_reg_decorr_dim_epk": train_loss_reg_decorr_dim_epk,
                "train_acc1_epk": train_acc1_epk,
                "train_acc5_epk": train_acc5_epk,

                "val_acc1_epk": val_acc1_epk,
                "val_acc5_epk": val_acc5_epk,
                "val_loss_cls_epk": val_loss_cls_epk,
                "val_loss_reg_orig_rad_epk": val_loss_reg_orig_rad_epk,
                "val_loss_reg_orig_dim_epk": val_loss_reg_orig_dim_epk,
                "val_loss_reg_decorr_rad_epk": val_loss_reg_decorr_rad_epk,
                "val_loss_reg_decorr_dim_epk": val_loss_reg_decorr_dim_epk,
                }, 
                f"{save_dir_name}_stats.pkl")


def train(train_loader, model, criterion, optimizer, epoch, device, args, man_info_retriev_idx):
    # switch to train mode
    model.train()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_classify = AverageMeter('Loss_classify', ':.4e', loss_alpha=args.alphas[0])
    losses_reg_orig_rad = AverageMeter('Loss_reg_orig_rad', ':.4e', loss_alpha=args.alphas[1])
    losses_reg_orig_dim = AverageMeter('Loss_reg_orig_dim', ':.4e', loss_alpha=args.alphas[2])
    losses_reg_decorr_rad = AverageMeter('Loss_reg_decorr_rad', ':.4e', loss_alpha=args.alphas[3])
    losses_reg_decorr_dim = AverageMeter('Loss_reg_decorr_dim', ':.4e', loss_alpha=args.alphas[4])
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, 
         losses_classify, losses_reg_orig_rad, losses_reg_orig_dim, losses_reg_decorr_rad, losses_reg_decorr_dim,
         top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # obtain the corresponding index of the manifold data
        index2select = [man_info_retriev_idx[t] for t in target]

        # move data to the same device as model
        index2select = torch.tensor(index2select, dtype=int).to(device)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # compute output
        neural_out, classification_out, neural_man_orig_space, neural_man_decorr = model(images, index2select)

        loss_classify, loss_orig_rad, loss_orig_dim, \
        loss_decorr_rad, loss_decorr_dim = criterion(classification_out, target, neural_man_orig_space, neural_man_decorr, index2select)
        
        loss = args.alphas[0] * loss_classify + \
               args.alphas[1] * loss_orig_rad + \
               args.alphas[2] * loss_orig_dim + \
               args.alphas[3] * loss_decorr_rad + \
               args.alphas[4] * loss_decorr_dim


        # measure accuracy
        acc1, acc5 = accuracy(classification_out, target, topk=(1, 5))

        # record progress
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        losses_classify.update(loss_classify.item(), images.size(0))
        losses_reg_orig_rad.update(loss_orig_rad.item(), images.size(0))
        losses_reg_orig_dim.update(loss_orig_dim.item(), images.size(0))
        losses_reg_decorr_rad.update(loss_decorr_rad.item(), images.size(0))
        losses_reg_decorr_dim.update(loss_decorr_dim.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 and get_rank() == 0:
            progress.display(i + 1)
            print_safe("")

    return losses_classify.avg, losses_reg_orig_rad.avg, losses_reg_orig_dim.avg, \
        losses_reg_decorr_rad.avg, losses_reg_decorr_dim.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, device, args, man_info_retriev_idx):
    # switch to evaluate mode
    model.eval()

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target, _) in enumerate(loader):
                i = base_progress + i

                # obtain the corresponding index of the manifold data
                index2select = [man_info_retriev_idx[t] for t in target]

                # move data to the same device as model
                index2select = torch.tensor(index2select, dtype=int).to(device)
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                neural_out, classification_out, neural_man_orig_space, neural_man_decorr = model(images, index2select)

                loss_classify, loss_orig_rad, loss_orig_dim, loss_decorr_rad, loss_decorr_dim = criterion(classification_out, target, 
                                                                                     neural_man_orig_space, neural_man_decorr, index2select)
                
                # loss_classify *= args.alphas[0]
                # loss_orig_space *= args.alphas[1]
                # loss_decorr_rad *= args.alphas[2]
                # loss_decorr_dim *= args.alphas[3]
                
                # measure accuracy and record loss
                acc1, acc5 = accuracy(classification_out, target, topk=(1, 5))
                losses_classify.update(loss_classify.item(), images.size(0))

                losses_reg_orig_rad.update(loss_orig_rad.item(), images.size(0))
                losses_reg_orig_dim.update(loss_orig_dim.item(), images.size(0))

                losses_reg_decorr_rad.update(loss_decorr_rad.item(), images.size(0))
                losses_reg_decorr_dim.update(loss_decorr_dim.item(), images.size(0))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses_classify = AverageMeter('Loss_classify', Summary.NONE, loss_alpha=args.alphas[0])

    losses_reg_orig_rad = AverageMeter('Loss_reg_orig_rad', ':.4e', loss_alpha=args.alphas[1])
    losses_reg_orig_dim = AverageMeter('Loss_reg_orig_dim', ':.4e', loss_alpha=args.alphas[2])

    losses_reg_decorr_rad = AverageMeter('Loss_reg_decorr_rad', ':.4e', loss_alpha=args.alphas[3])
    losses_reg_decorr_dim = AverageMeter('Loss_reg_decorr_dim', ':.4e', loss_alpha=args.alphas[4])

    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses_classify, losses_reg_orig_rad, losses_reg_orig_dim, losses_reg_decorr_rad, losses_reg_decorr_dim, top1, top5],
        prefix='Test: ')

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses_classify.all_reduce()

        losses_reg_orig_rad.all_reduce()
        losses_reg_orig_dim.all_reduce()

        losses_reg_decorr_rad.all_reduce()
        losses_reg_decorr_dim.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.test_workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if get_rank() == 0:
        progress.display_summary()

    return top1.avg, top5.avg, losses_classify.avg, \
           losses_reg_orig_rad.avg, losses_reg_orig_dim.avg, losses_reg_decorr_rad.avg, losses_reg_decorr_dim.avg


if __name__ == '__main__':
    main()




