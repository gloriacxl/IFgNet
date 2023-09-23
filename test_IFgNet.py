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
from sklearn.metrics import roc_auc_score, average_precision_score, auc
import torch.nn.functional as F
from typing import Dict, List, Tuple
import pandas as pd
from statistics import mean as stat_mean
from numpy import ndarray
from skimage import measure


size: int = 256
max_size: int = 512
mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

def mean_smoothing(amaps, kernel_size: int = 21) :

    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
    mean_kernel = mean_kernel.to(amaps.device)
    return F.conv2d(amaps, mean_kernel, padding=kernel_size // 2, groups=1)


def write_results_to_file(run_name, image_auc, pixel_auc):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)





def test(obj_names, mvtec_path, checkpoint_path, save_name_, now_epoch):
    obj_auroc_pixel_list = []
    obj_auroc_image_list = []
    for obj_name in obj_names:
        device = torch.device("cuda:3")
        img_dim = 256
        run_name_ = base_model_name + obj_name + '_' + 'a_' + str(args.a) + '_b_' + str(args.b) +'/'
        run_name_epoch = run_name_ + 'now_epoch_' + now_epoch + '/'
        
        save_name = save_name_

        model = ReconstructiveSubNetwork() # in_channels=3
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, save_name+".pckl"), map_location=device))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, save_name+"_seg.pckl"), map_location=device))
        model_seg.cuda()
        model_seg.eval()


        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size = 1,
                                shuffle=False, num_workers = args.num_workers)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        mask_cnt = 0

        anomaly_score_gt = []
        anomaly_score_prediction = []

        img_show_path = 'Test_img'+'/' + run_name_ + 'epoch_' + now_epoch +'/'
        if not os.path.exists(img_show_path):
            os.makedirs(img_show_path)


        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            with torch.no_grad():
                gray_rec, out_salient, _ = model(gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)
                out_salient_sm = torch.softmax(out_salient, dim=1)

            thre_list = [150,200,150,50,100,150,100,30,80,55,30,50] 
            picked_thre = thre_list[int(args.obj_id)]
                
            for i in range(gray_batch.shape[0]):
                mask_array = np.array((out_salient_sm[i,1:,...]*255).detach().squeeze(0).cpu())
                mask_array[mask_array <= picked_thre] = 0 
                mask_array[mask_array > picked_thre] = 255

                final_out_mask = out_salient_sm*out_mask_sm
                gray_batch_show = (gray_batch[i,...]*255).detach().cpu().numpy()
                final_out_mask_show = (final_out_mask[i,1:,...]*255).detach().cpu().numpy()
                true_mask_show = (true_mask[i,...]*255).detach().cpu().numpy()

                diff_ = out_mask_sm[0, 1, :, :].detach().cpu().numpy()
                diff_ = ((diff_ - diff_.min()) * 255 / (diff_.max() - diff_.min()))
                diff_ = cv2.applyColorMap(np.uint8(diff_), cv2.COLORMAP_JET)
                stacked_img = gray_batch_show.transpose(1, 2, 0)
                hot_img = cv2.addWeighted(np.uint8(diff_), 0.35, np.uint8(stacked_img), 0.5, 0)
                cv2.imwrite(os.path.join(img_show_path, '{}_loc.png'.format(i_batch*args.batch_size+i)),hot_img) 
                cv2.imwrite(os.path.join(img_show_path, '{}_ori.png'.format(i_batch*args.batch_size+i)), gray_batch_show.transpose([1, 2, 0]))
                cv2.imwrite(os.path.join(img_show_path, '{}_true_mask.png'.format(i_batch*args.batch_size+i)), true_mask_show.transpose([1, 2, 0]))

            out_mask_bs = torch.tensor(final_out_mask_show).unsqueeze(0) 

            out_mask_cv = out_mask_bs[0 ,: ,: ,:].detach().cpu().numpy()

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_bs[: ,: ,: ,:], 21, stride=1,
                                                               padding=21 // 2).cpu().detach().numpy()
            
           
            image_score = np.max(out_mask_averaged)

            anomaly_score_prediction.append(image_score)


            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        print(obj_name)
        print("AUC Image:  " +str(auroc))
        print("AUC Pixel:  " +str(auroc_pixel))
        print("==============================")

    print(run_name_)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))

    write_results_to_file(run_name_epoch, obj_auroc_image_list, obj_auroc_pixel_list)
  

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




if __name__ == "__main__":
    from argparse import ArgumentParser, Namespace
    import yaml

    parser = ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', action='store', type=float, default=0.0002)
    parser.add_argument('--epochs', action='store', type=int, default=500)
    parser.add_argument('--gpu_id', action='store', type=int, default=3)
    parser.add_argument('--log_path', action='store', type=str, default='./log/')
    parser.add_argument('--data_path', action='store', type=str, default=r'')   # dataset dir
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


    run_name_top = base_model_name + str(picked_classes[0]) + '_' + 'a_' + str(args.a) + '_b_' + str(args.b) +'/'
    run_name = run_name_top + 'IFgNet_' + str(picked_classes[0])
    save_name = run_name + "_epoch" + str(args.epochs)
    with torch.cuda.device(args.gpu_id):
        test(picked_classes, args.data_path, args.checkpoint_path, save_name, str(args.epochs))
