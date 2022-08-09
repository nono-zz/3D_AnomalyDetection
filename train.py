import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np

import torch.nn.MSELoss as MSEloss

from models.model_DRAEM import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
from torchvision import transforms
import tqdm

import torch.nn.functional as F
import random

import dataset.mvtec as mvtec
from dataset.mvtec import MVTecDataset


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def mean(list_x):
    return sum(list_x)/len(list_x)

def get_data_transforms(size, isize):
    # mean_train = [0.485]         # how do you set the mean_train and std_train in the get_data_transforms function?
    # mean_train = [-0.1]
    # std_train = [0.229]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        
        #transforms.CenterCrop(args.input_size),
        transforms.ToTensor()
        # transforms.Normalize(mean=mean_train,
        #                      std=std_train)
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

        
        
        
def DRAEM_train(args, dataloader, model, model_seg, loss_l2, loss_ssim, loss_focal, optimizer, visualizer, scheduler, run_name):
    class_names = mvtec.CLASS_NAMES if args.class_name == 'all' else [args.class_name]
    device = args.device

    for class_name in class_names:
        best_img_roc = -1
        best_pxl_roc = -1
        best_pxl_pro = -1
        print(' ')
        print('%s | newly initialized...' % class_name)
        
        
    train_dataset    = MVTecDataset(dataset_path  = args.data_path, 
                                    class_name    =     class_name, 
                                    resize        =            256,
                                    cropsize      =      args.size,
                                    is_train      =           True,
                                    wild_ver      =        args.Rd)
        
    test_dataset     = MVTecDataset(dataset_path  = args.data_path, 
                                    class_name    =     class_name, 
                                    resize        =            256,
                                    cropsize      =      args.size,
                                    is_train      =          False,
                                    wild_ver      =        args.Rd)


    train_loader   = DataLoader(dataset         = train_dataset, 
                                batch_size      =             1, 
                                pin_memory      =          True,
                                shuffle         =          True,
                                drop_last       =          True,)

    test_loader   =  DataLoader(dataset        =   test_dataset, 
                                batch_size     =              1, 
                                pin_memory     =           True,)
    
    
    if args.cnn == 'wrn50_2':
        model = wrn50_2(pretrained=True, progress=True)
    elif args.cnn == 'res18':
        model = res18(pretrained=True,  progress=True)
    elif args.cnn == 'effnet-b5':
        model = effnet.from_pretrained('efficientnet-b5')
    elif args.cnn == 'vgg19':
        model = vgg19(pretrained=True, progress=True)
    elif args.cnn == 'DRAEM':
        rgb2depthModel = DiscriminativeSubNetwork(in_channels=3, out_channels=1)
        depth2rgbModel = DiscriminativeSubNetwork(in_channels=1, out_channels=3)
        
    rgb2depthModel, depth2rgbModel = rgb2depthModel.to(device), depth2rgbModel.to(device)
        
    loss_MSE = torch.nn.MSELoss()
    learning_rate = args.lr
    optimizer = torch.optim.Adam(list(rgb2depthModel.parameters())+list(depth2rgbModel.parameters()), lr=learning_rate)
    
    for epoch in tqdm(range(args.epochs), '%s -->'%(class_name)):

        rgb2depthModel.train()
        depth2rgbModel.train()
        
        loss_list = []
        for (rgb, pc, depth, label, mask) in train_loader:
            
            
            depthPred = rgb2depthModel(rgb.to(device))
            rgbPred = depth2rgbModel(depth.to(device))
            
            loss = loss_MSE(rgb, rgbPred) + loss_MSE(depth, depthPred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
        
        print('loss is:   ', np.mean(loss_list))
        print('the radium of loss function is: ', loss_fn.r.item())

        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
        torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))
        
                
        
        

if __name__=="__main__":
    
    # python train_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path 
    # --anomaly_source_path  --checkpoint_path ./checkpoints/ --log_path ./logs/
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', default=1,  action='store', type=int)
    parser.add_argument('--bs', default = 8, action='store', type=int)
    parser.add_argument('--lr', default=0.0001, action='store', type=float)
    parser.add_argument('--epochs', default=700, action='store', type=int)
    parser.add_argument('--gpu_id', default=0, action='store', type=int, required=False)
    parser.add_argument('--data_path', default='/home/zhaoxiang/baselines/DRAEM/datasets/mvtec/', action='store', type=str)
    parser.add_argument('--anomaly_source_path', default='/home/zhaoxiang/baselines/DRAEM/datasets/dtd/images/', action='store', type=str)
    parser.add_argument('--checkpoint_path', default='./checkpoints/', action='store', type=str)
    parser.add_argument('--log_path', default='./logs/', action='store', type=str)
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--experiment_name', default='liver', choices=['retina, liver, brain, head'], action='store')
    parser.add_argument('--backbone', default='noise', action='store')
    
    # for noise autoencoder
    parser.add_argument("-nr", "--noise_res", type=float, default=16,  help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument("-img_size", "--img_size", type=float, default=256, help="noise magnitude.")
    parser.add_argument('--dataset_name', default='hist_DIY', choices=['hist_DIY', 'Brain_MRI', 'CovidX', 'RESC_average'], action='store')
    parser.add_argument('--model', default='DRAEM_discriminitive', choices=['ws_skip_connection', 'DRAEM_reconstruction', 'DRAEM_discriminitive'], action='store')
    parser.add_argument('--process_method', default='Guassian_noise', choices=['none', 'Guassian_noise', 'DRAEM'], action='store')
    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        DRAEM_train(args)
