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
    
    
    # print('epoch {}, loss is: {} ',format(epoch, np.mean(loss_list)))

    # if (epoch+1) % 3 == 0:
    #     current_path = os.path.dirname(os.path.abspath(__file__))
        
    #     checkpoint_path = current_path + args.checkpoint_path
    #     if not os.path.exists(checkpoint_path):
    #         os.mkdir(checkpoint_path)
            
    #     torch.save(rgb2depthModel.state_dict(), os.path.join(checkpoint_path, "rgb2depthModel.pckl"))
    #     torch.save(depth2rgbModel.state_dict(), os.path.join(checkpoint_path, "depth2rgbModel.pckl"))
        
                

        

