
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
num_categories_dict = {2: "toy", # test toy dataset
                       50: "textshape50",
                       209: "human16-209",
                       16: "",
                       1000: ""
                       }

parser = argparse.ArgumentParser(description='PyTorch imagenet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--save-dir', required=True, type=str, help='path to save the checkpoints')
parser.add_argument("--save-suffix", default=None, type=str, help="special suffix for the save directory")


parser.add_argument('--img-folder-txt', default="./data/textshape50.txt",
                    type=str, help='path to a textfile of sub categories of imagenet to be used')

parser.add_argument('--num-category', default=50, type=int, 
                    help=f'number of categories to use, must be one of {list(num_categories_dict.keys())}')
# parser.add_argument('--category-209', action='store_true', default=True,
#                     help='use the 209 fine-grained categories belonged the 16 basic-level categories')
# parser.add_argument('--category-209', action='store_true', default=True,
#                     help='use the 209 fine-grained categories belonged the 16 basic-level categories')
# parser.add_argument('--category-16', action='store_true', help='use the 16 basic-level categories')
# parser.add_argument('--category-1k', action='store_true', help='use the original 1k categories')

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
parser.add_argument("--custom-sigma", default=None, type=float, help="custom sigma for the bandpass layer")

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

    # if args.category_16 or args.category_1k:
    #     args.category_209 = False

    assert args.num_category in num_categories_dict, f"num_category must be one of {list(num_categories_dict.keys())}"
    assert num_categories_dict[args.num_category] in args.img_folder_txt, \
        f"num_category {args.num_category} does not match the txt file {args.img_folder_txt}"

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
    # category_str = "209" if args.category_209 else "16" if args.category_16 else "1k"
    save_dir_name = f"{args.save_dir}/{args.arch}-layer-{args.append_layer}-category-{args.num_category}"
    if args.save_suffix:
        save_dir_name += f"-{args.save_suffix}"
    save_dir_name += f"-{datetime_str}"

    utils.print_safe(f"*** Saving to: {save_dir_name}")
    utils.make_directory(save_dir_name)
    
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
    
    # for i, (images, target) in enumerate(train_loader):
    #     print(i, images.shape, target)

    utils.print_safe(f"Data loaded: train: {len(train_loader)}, val: {len(val_loader)}", flush=True)

    ## -- create model
    # if args.category_16:
    #     num_classes = 16
    # elif args.category_1k:
    #     num_classes = 1000
    # else:
    #     num_classes = 209
    classifier = models.get_classifier(args.arch, num_classes=args.num_category)

    if args.append_layer == "bandpass":
        model = models.BandPassNet(classifier, kernel_size=args.kernel_size, custom_sigma=args.custom_sigma)
    elif args.append_layer == "blur":
        #TODO: Implement blur layer
        raise NotImplementedError
        # model = models.BlurNet(classifier)
    else:
        model = classifier
    utils.print_safe(model)

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
                utils.print_safe(f"distributing batch size: {args.batch_size}")
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
    criterion = models.utils.get_loss_fn().to(device)

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
            utils.print_safe("=> loading checkpoint '{}'".format(args.resume))
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
            utils.print_safe("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            utils.print_safe("=> no checkpoint found at '{}'".format(args.resume))


    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    train_loss_cls_epk = []
    train_acc1_epk = []
    train_acc5_epk = []

    val_loss_cls_epk = []
    val_acc1_epk = []
    val_acc5_epk = []

    for epoch in range(args.start_epoch, args.epochs):
        utils.print_safe(f"Epoch: {epoch}, lr: {scheduler.get_last_lr()}, {optimizer.param_groups[0]['lr']}")
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss_classify, train_top1, train_top5 = train(
            train_loader, model, criterion, optimizer, epoch, device, args
            )

        # evaluate on validation set
        val_loss_classify, acc1, acc5 = validate(
            val_loader, model, criterion, device, args
            )

        train_loss_cls_epk.append(train_loss_classify)
        train_acc1_epk.append(train_top1)
        train_acc5_epk.append(train_top5)

        val_loss_cls_epk.append(val_loss_classify)
        val_acc1_epk.append(acc1)
        val_acc5_epk.append(acc5)
        

        scheduler.step()

        if (
                epoch % args.save_interval == 0 and
                epoch != 0 and
                utils.is_main_process()
                # args.multiprocessing_distributed and
                # args.rank % ngpus_per_node == 0
        ):
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict() if args.multiprocessing_distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, f"{save_dir_name}/ckpt_epk{epoch}.pth")

            utils.pickle_dump({
                "train_loss_cls_epk": train_loss_cls_epk,
                "train_acc1_epk": train_acc1_epk,
                "train_acc5_epk": train_acc5_epk,
                "val_loss_cls_epk": val_loss_cls_epk,
                "val_acc1_epk": val_acc1_epk,
                "val_acc5_epk": val_acc5_epk,
                }, 
                f"{save_dir_name}/stats_epk{epoch}.pkl")


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    # switch to train mode
    model.train()

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses_classify = utils.AverageMeter('Loss_classify', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, 
         losses_classify,
         top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # compute output
        output = model(images)

        # measure accuracy
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # loss
        loss = criterion(output, target) 
        losses_classify.update(loss.item(), images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 and utils.is_main_process():
            progress.display(i + 1)
            utils.print_safe("")

    return losses_classify.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, device, args):
    # switch to evaluate mode
    model.eval()

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                # move data to the same device as model
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                output = model(images)
                
                # measure accuracy and record loss
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # loss
                loss = criterion(output, target)
                losses_classify.update(loss.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

    batch_time = utils.AverageMeter('Time', ':6.3f', utils.Summary.NONE)
    losses_classify = utils.AverageMeter('Loss_classify', utils.Summary.NONE)

    top1 = utils.AverageMeter('Acc@1', ':6.2f', utils.Summary.AVERAGE)
    top5 = utils.AverageMeter('Acc@5', ':6.2f', utils.Summary.AVERAGE)

    progress = utils.ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses_classify, top1, top5],
        prefix='TEST: ')

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        losses_classify.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.test_workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if utils.is_main_process():
        progress.display_summary()

    return losses_classify.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()




