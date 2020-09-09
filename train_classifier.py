import sys
import argparse
import json
import os
import pdb
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torch.distributed as dist
from losses import RegLoss

from dataset import LandmarkDataset, collate_function
from tools.config import load_config
from tools.schedulers import create_optimizer
from tools.utils import read_annotations, get_train_paths, logger
from tools.transforms import create_train_transforms,create_val_transforms
from models import classifier
from apex import amp
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from common import DFL_DATA_PATH, DFL_CONFIG, DFL_DEVICE
from tensorboardX import SummaryWriter
from tools.train import validate, train_epoch

import warnings
warnings.filterwarnings("ignore")


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--data_path', type=str, default=DFL_DATA_PATH,help='path to datasets. (default: %s)' % DFL_DATA_PATH)
    parser.add_argument('--config_name', metavar='CONFIG_FILE',default=DFL_CONFIG,help='path to configuration file')
    parser.add_argument('--train_collection', type=str, help='training collection',required=True)
    parser.add_argument('--val_collection', type=str, help='validation collection',required=True)
    parser.add_argument('--run_id', default=DFL_DEVICE, type=int, help='run_id (default: 0)')
    parser.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    parser.add_argument('--workers', type=int, default=6, help='number of cpu threads to use')
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--zero_score', action='store_true', default=False)
    parser.add_argument('--from_zero', action='store_true', default=True)
    parser.add_argument("--seed", default=777, type=int)
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=4))
    conf = load_config("configs/%s"%args.config_name)


    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True

    model_dir, train_data_path, val_data_path = get_train_paths(args)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    train_params = {
        'data_path':args.data_path,
        'train_collection':args.train_collection,
        'val_collection':args.val_collection,
        'config_name':args.config_name,
        'run_id':args.run_id,
        'prefix':args.prefix
    }
    train_params_file = os.path.join(model_dir, 'train_params.json')
    with open(train_params_file, 'w') as fp:
        json.dump(train_params, fp, indent=4)
    #os.makedirs(model_dir, exist_ok=True)
    for data_path in [train_data_path, val_data_path]:
        if not os.path.exists(data_path):
            logger.error("{} does not exist".format(data_path))
            sys.exit(0)


    max_epochs = conf['optimizer']['schedule']['epochs']
    bce_best = 100
    start_epoch = 0

    if args.debug:
        logger.info("debug mode")
        pdb.set_trace()


    """
    dataset & dataloader
    """
    batch_size = conf['optimizer']['batch_size']
    data_train = LandmarkDataset(
        annotations=read_annotations(train_data_path),
        mode="train",
        transforms=create_train_transforms(conf["size"]),
        normalize=conf.get("normalize", None))
    data_val = LandmarkDataset(
        annotations=read_annotations(val_data_path),
        mode="val",
        transforms=create_val_transforms(conf["size"]),
        normalize=conf.get("normalize", None))

    val_data_loader = DataLoader(
        data_val,
        batch_size=batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_function)


    """
    model 定义、加载 & loss_func、优化器
    """
    model = classifier.__dict__[conf['network']](encoder=conf['encoder'])
    model = model.cuda()
    if args.distributed:
        model = convert_syncbn_model(model)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
                if not args.zero_score:
                    bce_best = checkpoint.get('bce_best', 0)
            print("=> loaded checkpoint '{}' (epoch {}, bce_best {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['bce_best']))
        else:
            logger.warning("=> no checkpoint found at '{}', train from begining".format(args.resume))

    loss_function = RegLoss().cuda()

    if conf["optimizer"]["schedule"]["mode"]  == "step":
        temp = conf['optimizer']
        temp["schedule"]['params']['max_iter'] = len(data_train) // batch_size
        optimizer, scheduler = create_optimizer(temp, model)
    elif conf["optimizer"]["schedule"]["mode"]  == "epoch":
        optimizer, scheduler = create_optimizer(conf["optimizer"], model)


    """
    混合精度 & 多卡训练
    """
    if conf['fp16']:
        model, optimizer = amp.initialize(model, optimizer, opt_level=conf['opt_level'], loss_scale='dynamic')
    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True,find_unused_parameters=True)
    else:
        model = DataParallel(model).cuda()


    summary_writer = SummaryWriter(log_dir=model_dir)
    current_epoch = start_epoch
    data_val.reset_seed(1, args.seed)


    for epoch in range(start_epoch, max_epochs):
        data_train.reset_seed(epoch, args.seed)
        train_sampler = None
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
            train_sampler.set_epoch(epoch)
        # 前freeze_epoch轮不参与训练
        if epoch < conf['freeze_epochs']:
            logger.info("Freezing encoder!!!")
            model.module.encoder.eval()
            for p in model.module.encoder.parameters():
                p.requires_grad = False
        else:
            model.module.encoder.train()
            for p in model.module.encoder.parameters():
                p.requires_grad = True

        train_data_loader = DataLoader(
            data_train,
            batch_size=batch_size,
            num_workers=args.workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_function)

        train_epoch(current_epoch, loss_function, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                    args.local_rank, args.debug)

        model = model.eval()
        if args.local_rank == 0:
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': bce_best,
            }, model_dir + '/model_last.pth.tar')
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'bce_best': bce_best,
            }, model_dir + "/model_{}.pth.tar".format(current_epoch))

            if (epoch + 1) % conf['val_freq'] == 0:
                bce_best = validate(args, val_data_loader, bce_best, model,
                                        model_dir = model_dir,
                                        current_epoch=current_epoch,
                                        summary_writer=summary_writer,
                                        conf=conf)
        current_epoch += 1


if __name__ == '__main__':
    main()
