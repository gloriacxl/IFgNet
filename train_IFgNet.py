import torch
import time
import datetime
from data_loader import MVTecDRAEMTrainDataset,MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
from torch import optim
from model_mtas_d import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from random import choice
import cv2
import matplotlib.pyplot as plt
import pylab
import random
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F
from skimage import exposure
from typing import Dict, List, Tuple

size: int = 256
max_size: int = 512
mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def __len__(self):
        return len(self.image_paths)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(obj_names, args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)


    for obj_name in obj_names:
        run_name_top = base_model_name + obj_name + '_' + 'a_' + str(args.a) + '_b_' + str(args.b) +'/'
        if not os.path.exists(os.path.join(args.checkpoint_path, run_name_top)):
             os.makedirs(os.path.join(args.checkpoint_path, run_name_top))

        run_name = run_name_top + 'IFgNet_' + obj_name
        writer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

        model = ReconstructiveSubNetwork() # in_channels=3
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)

        optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": args.lr},
            {"params": model_seg.parameters(), "lr": args.lr}])

        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecDRAEMTrainDataset(args.data_path + obj_name + "/train/", args.anomaly_source_path,
                                         resize_shape=[256, 256])

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

        prev_time = time.time()


        l2_running_loss = 0.0
        ssim_running_loss = 0.0
        segment_running_loss = 0.0
        salient_running_loss = 0.0
        total_running_loss = 0.0



        for epoch in range(args.epochs):
            l2_loss_of_epoch = 0
            ssim_loss_of_epoch = 0
            segment_loss_of_epoch = 0
            salient_loss_of_epoch = 0
            loss_of_epoch = 0


            for i_batch, sample_batched in enumerate(dataloader):
                gray_batch = sample_batched["image"].cuda()
                aug_gray_batch = sample_batched["augmented_image"].cuda()
                anomaly_mask = sample_batched["anomaly_mask"].cuda()
                foreground_batch = sample_batched["foreground_image"].cuda()
                foreground_batch_new = sample_batched["foreground_image_new"].cuda()

                optimizer.zero_grad()
                
                gray_rec, out_salient, _ = model(aug_gray_batch)
                joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)
                out_salient_sm = torch.softmax(out_salient, dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                l2_loss = loss_l2(gray_rec,gray_batch)
                ssim_loss = loss_ssim(gray_rec, gray_batch)
                

                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                salient_loss = loss_focal(out_salient_sm, foreground_batch_new)
                loss = l2_loss + ssim_loss + segment_loss + salient_loss

                l2_loss_of_epoch += l2_loss.item()
                ssim_loss_of_epoch += ssim_loss.item()
                segment_loss_of_epoch += segment_loss.item()
                salient_loss_of_epoch += salient_loss.item()
                loss_of_epoch += loss.item()

                loss.backward()
                optimizer.step()


                l2_loss_of_epoch += l2_loss.item()
                ssim_loss_of_epoch += ssim_loss.item()
                segment_loss_of_epoch += segment_loss.item()
                salient_loss_of_epoch += salient_loss.item()
                loss_of_epoch += loss.item()

                l2_running_loss += l2_loss.item()
                ssim_running_loss += ssim_loss.item()
                segment_running_loss += segment_loss.item()
                salient_running_loss += salient_loss.item()
                total_running_loss += loss.item()


                if (epoch * len(dataloader) + i_batch) % 50 == 49:  
                    writer.add_scalar('loss/l2_loss', l2_running_loss / 50, epoch * len(dataloader) + i_batch)
                    writer.add_scalar('loss/ssim_loss', ssim_running_loss / 50, epoch * len(dataloader) + i_batch)
                    writer.add_scalar('loss/segment_loss', segment_running_loss / 50,
                                      epoch * len(dataloader) + i_batch)
                    writer.add_scalar('loss/salient_loss', salient_running_loss / 50,
                                      epoch * len(dataloader) + i_batch)
                    writer.add_scalar('loss/total_loss', total_running_loss / 50, epoch * len(dataloader) + i_batch)
                    l2_running_loss = 0.0
                    ssim_running_loss = 0.0
                    segment_running_loss = 0.0
                    salient_running_loss = 0.0
                    total_running_loss = 0.0

                batches_done = epoch * len(dataloader) + i_batch
                batches_left = args.epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(

                    "\r[%s][Epoch %d/%d][Batch %d/%d][l2_loss: %f, ssim_loss: %f, segment_loss: %f, salient_loss: %f][loss: %f] ETA: %s"
                    % (
                        obj_name,
                        epoch + 1,
                        args.epochs,
                        i_batch,
                        len(dataloader),
                        l2_loss.item(),
                        ssim_loss.item(),
                        segment_loss.item(),
                        salient_loss.item(),
                        loss.item(),
                        time_left

                    )
                )

            print(
                "\r[%s][Epoch %d/%d] [Batch %d/%d] [l2_loss_all: %f, ssim_loss_all: %f, segment_loss_all: %f, salient_loss_all: %f][loss_all: %f] ETA: %s"
                % (
                    obj_name,
                    epoch + 1,
                    args.epochs,
                    len(dataloader),
                    len(dataloader),
                    l2_loss_of_epoch,
                    ssim_loss_of_epoch,
                    segment_loss_of_epoch,
                    salient_loss_of_epoch,
                    loss_of_epoch,
                    time_left
                )
            )


            if args.checkpoint_interval != -1 and (epoch + 1) % args.checkpoint_interval == 0: 
                save_name = run_name + "_epoch" + str(epoch + 1)
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, save_name + ".pckl"))
                torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, save_name + "_seg.pckl"))
            

if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace
    import yaml

    parser = ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', action='store', type=float, default=0.0002)
    parser.add_argument('--epochs', action='store', type=int, default=500)
    parser.add_argument('--gpu_id', action='store', type=int, default=2)
    parser.add_argument('--log_path', action='store', type=str, default='./log/')
    parser.add_argument('--data_path', action='store', type=str, default=r'')   # dataset dir
    parser.add_argument('--anomaly_source_path', action='store', default=r'', type=str) # dtd dataset dir
    parser.add_argument('--checkpoint_path', action='store', type=str, default='./checkpoint/')
    parser.add_argument('--checkpoint_interval', type=int, default=500)  
    parser.add_argument('--image_visual_interval', type=int, default=100)  
    parser.add_argument('--a', type=float, default=1)  
    parser.add_argument('--b', type=float, default=1)  
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--suffix", type=str, default='')
    
    args = parser.parse_args()

    base_model_name = 'IFgNet'+'_lr' + str(args.lr) +'_epochs'+str(args.epochs)+'_bs'+str(args.batch_size)+'_'+'VisA'+str(args.obj_id) + '/'

    obj_batch = [['candle'],        #0
                 ['capsules'],      #1
                 ['cashew'],        #2
                 ['chewinggum'],    #3
                 ['fryum'],         #4
                 ['macaroni1'],     #5
                 ['macaroni2'],     #6
                 ['pcb1'],          #7
                 ['pcb2'],          #8
                 ['pcb3'],          #9
                 ['pcb4'],          #10
                 ['pipe_fryum']     #11
                 ]

    picked_classes = obj_batch[int(args.obj_id)]

    with torch.cuda.device(args.gpu_id):
        train_on_device(picked_classes, args)
