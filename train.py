import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
import tqdm

import dataset.mvtec as mvtec
from dataset.mvtec import MVTecDataset
from models.model_DRAEM import ReconstructiveSubNetwork, DiscriminativeSubNetwork



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

        
        
        
def DRAEM_train(args):
    class_names = mvtec.CLASS_NAMES if args.class_name == 'all' else [args.class_name]
    device = torch.device('cuda:{}'.format(args.gpu_id))

    for class_name in class_names:

        print(' ')
        print('%s | newly initialized...' % class_name)
        
        
        train_dataset    = MVTecDataset(dataset_path  = args.data_path, 
                                        class_name    =     class_name, 
                                        resize        =            256,
                                        is_train      =           True)
            
        test_dataset     = MVTecDataset(dataset_path  = args.data_path, 
                                        class_name    =     class_name, 
                                        resize        =            256,
                                        is_train      =          False)


        train_loader   = DataLoader(dataset         = train_dataset, 
                                    batch_size      =             1, 
                                    pin_memory      =          True,
                                    shuffle         =          True,
                                    drop_last       =          True,)

        test_loader   =  DataLoader(dataset        =   test_dataset, 
                                    batch_size     =              1, 
                                    pin_memory     =           True,)
        
        
        if args.backbone == 'wrn50_2':
            model = wrn50_2(pretrained=True, progress=True)
        elif args.backbone == 'res18':
            model = res18(pretrained=True,  progress=True)
        elif args.backbone == 'effnet-b5':
            model = effnet.from_pretrained('efficientnet-b5')
        elif args.backbone == 'vgg19':
            model = vgg19(pretrained=True, progress=True)
        elif args.backbone == 'DRAEM':
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
            
            print('epoch[{}/{}],  loss is: {} ',format(epoch, args.epochs, np.mean(loss_list)))

            torch.save(rgb2depthModel.state_dict(), os.path.join(args.checkpoint_path, "rgb2depthModel.pckl"))
            torch.save(depth2rgbModel.state_dict(), os.path.join(args.checkpoint_path, "depth2rgbModel.pckl"))
        
                
        
        

if __name__=="__main__":
    
    # python train_DRAEM.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path 
    # --anomaly_source_path  --checkpoint_path ./checkpoints/ --log_path ./logs/
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default = 8, action='store', type=int)
    parser.add_argument('--lr', default=0.001, action='store', type=float)
    parser.add_argument('--epochs', default=80, action='store', type=int)
    parser.add_argument('--gpu_id', default=0, action='store', type=int, required=False)
    parser.add_argument('--data_path', default='/home/zhaoxiang/3D-ADS/datasets/mvtec3d', type=str)
    parser.add_argument('--checkpoint_path', default='./checkpoints/', action='store', type=str)
    parser.add_argument('--backbone', default='DRAEM', action='store')
    parser.add_argument('--class_name', default='all', action='store')
    
    
    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        DRAEM_train(args)
