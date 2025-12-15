import os
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.dataset import MedicalDataSets
import albumentations as A
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
import logging
import sys
from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score
from tensorboardX import SummaryWriter
from network.GLSRUNet import glsrunet


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch(3407)

parser = argparse.ArgumentParser()

# BUSI
parser.add_argument('--model', type=str, default="glsrunet",
                    choices=["glsrunet"], help='model')
parser.add_argument('--base_dir', type=str, default="Datasets/busi", help='dir')
parser.add_argument('--train_file_dir', type=str, default="busi_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="busi_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')


args = parser.parse_args()


def getDataloader():
    img_size = 256
    train_transform = Compose([
        RandomRotate90(),
        A.HorizontalFlip(),
        Resize(img_size, img_size),
        A.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        A.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train", transform=train_transform,
                               train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                             train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=8, shuffle=True,
                             num_workers=4, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    return trainloader, valloader


def get_model(args):
    if args.model == "glsrunet":
        model = glsrunet()
    else:
        model = None
        print("model err")
    return model.cuda()


def train(args):
    base_lr = args.base_lr
    trainloader, valloader = getDataloader()
    model = get_model(args)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()
    best_iou = 0
    iter_num = 0
    max_epoch = 300
    max_iterations = len(trainloader) * max_epoch
    args.snapshot_path = f'log/{args.train_file_dir.split("_")[0]}/{args.model}/{args.batch_size}bs_{max_epoch}eps_{base_lr}lr'
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    logging.basicConfig(filename=args.snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(args.snapshot_path + '/log')
    logging.info(f'{len(trainloader)} iterations per epoch')
    for epoch_num in range(max_epoch):
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'SE': AverageMeter(),
                      'PC': AverageMeter(),
                      'F1': AverageMeter(),
                      'ACC': AverageMeter()
                      }
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            loss = criterion(outputs, label_batch)
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))
            writer.add_scalar('train_info/train_loss', avg_meters['loss'].avg, epoch_num)
            writer.add_scalar('train_info/train_iou', avg_meters['iou'].avg, epoch_num)

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                
                iou, _, SE, PC, F1, _, ACC = iou_score(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['SE'].update(SE, input.size(0))
                avg_meters['PC'].update(PC, input.size(0))
                avg_meters['F1'].update(F1, input.size(0))
                avg_meters['ACC'].update(ACC, input.size(0))
                writer.add_scalar('val_info/val_loss', avg_meters['val_loss'].avg, epoch_num)
                writer.add_scalar('val_info/val_iou', avg_meters['val_iou'].avg, epoch_num)
                writer.add_scalar('val_info/val_SE', avg_meters['SE'].avg, epoch_num)
                writer.add_scalar('val_info/val_PC', avg_meters['PC'].avg, epoch_num)
                writer.add_scalar('val_info/val_F1', avg_meters['F1'].avg, epoch_num)
                writer.add_scalar('val_info/val_ACC', avg_meters['ACC'].avg, epoch_num)

        logging.info(
            f'epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f '
            '- val_loss %.4f - val_iou %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
            avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['SE'].avg,
            avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg))

        if avg_meters['val_iou'].avg > best_iou:
            if not os.path.exists('./checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(), 'checkpoint/{}_model_{}.pth'
                       .format(args.model, args.train_file_dir.split(".")[0]))
            best_iou = avg_meters['val_iou'].avg
            logging.info("=> saved best model")
    logging.info(f'best iou: {best_iou}')
    logging.info(f'================================================================================')
    return "Training Finished!"


if __name__ == "__main__":
    train(args)
