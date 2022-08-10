import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
import torchvision
from tqdm import tqdm
import pathlib

import dataset.mvtec as mvtec
from dataset.mvtec import MVTecDataset, MVTec_RGBD_Dataset
from models.model_DRAEM import ReconstructiveSubNetwork, DiscriminativeSubNetwork

# from torch.utils.tensorboard import SummaryWriter
from utils.tensorboard_visualizer import TensorboardVisualizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def mean(list_x):
    return sum(list_x)/len(list_x)
   
def test(args, model, test_loader, class_name, visualizer, device, lossFN, epoch):

    print(' ')
    print('%s | begin testing...' % class_name)

    model.eval()
    
    error_list, mseError_list, cosError_list = [], [], []
    
    for (rgb, depth, mask, label) in test_loader:
        rgb = rgb.to(device)
        depth = depth.to(device)
        
        rgbPred, depthPred = model(rgb, depth)
        
        if args.loss_mode == 'MSE_Cos':
            
            error, mseError, cosError = lossFN(rgb, depth, rgbPred, depthPred)
            mseError_list.append(mseError.item())
            cosError_list.append(cosError.item())
                
        else:
            error = lossFN(rgb, depth, rgbPred, depthPred)
            
        
        error_list.append(error.item())
        
        visualizer.plot_loss(mean(error_list), epoch, loss_name='error')
        visualizer.visualize_image_batch(rgb, epoch, image_name='rgb_raw_test')
        visualizer.visualize_image_batch(rgbPred, epoch, image_name='rgb_pred_test')
        visualizer.visualize_image_batch(depth, epoch, image_name='depth_raw_test')
        visualizer.visualize_image_batch(depthPred, epoch, image_name='depth_pred_test')
        
    if args.loss_mode == 'MSE_Cos':
        print('test: epoch [{}/{}], tot_error:{:.4f}, mseError:{:.4f}, cosError:{:.4f}'.format(epoch, args.epochs, np.mean(error_list), np.mean(mseError_list), np.mean(cosError_list)))
    else:
        print('test: epoch [{}/{}], tot_error:{:.4f}'.format(epoch, args.epochs, np.mean(error_list)))
    
    
    
def evaluation():
    anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
    # anomaly_map = gaussian_filter(anomaly_map, sigma=4)         # why use gaussian filter to blur the anomaly map?
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0
    if label.item()!=0:
        aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                        anomaly_map[np.newaxis,:,:]))
    gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
    pr_list_px.extend(anomaly_map.ravel())
    gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
    pr_list_sp.append(np.max(anomaly_map))
        

        

