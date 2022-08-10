import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from tqdm import tqdm
import dataset.mvtec as mvtec
from dataset.mvtec import MVTecDataset, MVTec_RGBD_Dataset

# models
from models.model_DRAEM import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from models.model_unet_skip_connection import CrossPredictionNetwork, RgbDepthNetwork

# loss
from loss.loss import Loss

# from torch.utils.tensorboard import SummaryWriter
from utils.tensorboard_visualizer import TensorboardVisualizer

from evaluation import test

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
    
    
    # set the tensorboard writer
    # log_dir = os.path.dirname(os.path.abspath(__file__)) + '/log'
    log_dir = '/home/zhaoxiang/log'
    
    visualizer = TensorboardVisualizer(log_dir=os.path.join(log_dir +"/"))
    for class_name in class_names:

        print(' ')
        print('%s | newly initialized...' % class_name)
        
        
        train_dataset    = MVTec_RGBD_Dataset(dataset_path  = args.data_path, 
                                        class_name    =     class_name, 
                                        resize        =            256,
                                        is_train      =           True)
            
        test_dataset     = MVTec_RGBD_Dataset(dataset_path  = args.data_path, 
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
        elif args.backbone == 'crossPred':
            model = CrossPredictionNetwork()
        elif args.backbone == 'DRAEM':
            model = RgbDepthNetwork()
            
        model = model.to(device)
        
        
        lossFN = Loss(args.loss_mode)
        learning_rate = args.lr
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        
        
        # save results
        
        epochs = args.epochs
        for epoch in tqdm(range(epochs), '%s -->'%(class_name)):

            model.train()
            
            loss_list, mseLoss_list, cosLoss_list = [], [], []
            for (rgb, depth, mask, label) in train_loader:
                rgb = rgb.to(device)
                depth = depth.to(device)
            
                rgbPred, depthPred = model(rgb, depth)
                
                # mseLoss = loss_MSE(rgb, rgbPred) + loss_MSE(depth, depthPred)
                # cosLoss = torch.mean(1 - loss_Cos(rgb, rgbPred)) + torch.mean(1 - loss_Cos(depth, depthPred))
                if args.loss_mode == 'MSE_Cos':
                    loss, mseLoss, cosLoss = lossFN(rgb, depth, rgbPred, depthPred)
                    mseLoss_list.append(mseLoss.item())
                    cosLoss_list.append(cosLoss.item())
                else:
                    loss = lossFN(rgb, depth, rgbPred, depthPred)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                
                visualizer.plot_loss(mean(loss_list), epoch, loss_name='loss')
                visualizer.visualize_image_batch(rgb, epoch, image_name='rgb_raw')
                visualizer.visualize_image_batch(rgbPred, epoch, image_name='rgb_pred')
                visualizer.visualize_image_batch(depth, epoch, image_name='depth_raw')
                visualizer.visualize_image_batch(depthPred, epoch, image_name='depth_pred')
                
            if args.loss_mode == 'MSE_Cos':
                print('epoch [{}/{}], tot_loss:{:.4f}, mseLoss:{:.4f}, cosLoss:{:.4f}'.format(epoch, args.epochs, np.mean(loss_list), np.mean(mseLoss_list), np.mean(cosLoss_list)))
            
            else:
                print('epoch [{}/{}], loss is: {} '.format(epoch, args.epochs, np.mean(loss_list)))
            
            
            test(args, model, test_loader, class_name, visualizer, device, lossFN, epoch)
    
            if (epoch+1) % 3 == 0:
                # current_path = os.path.dirname(os.path.abspath(__file__))
                current_path = '/home/zhaoxiang'                
                checkpoint_path = current_path + args.checkpoint_path
                if not os.path.exists(checkpoint_path):
                    os.mkdir(checkpoint_path)
            
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pckl"))
        
                
        

if __name__=="__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default = 8, action='store', type=int)
    parser.add_argument('--lr', default=0.001, action='store', type=float)
    parser.add_argument('--epochs', default=80, action='store', type=int)
    parser.add_argument('--gpu_id', default=0, action='store', type=int, required=False)
    parser.add_argument('--data_path', default='/home/zhaoxiang/3D-ADS/datasets/mvtec3d', type=str)
    parser.add_argument('--checkpoint_path', default='/checkpoints/', action='store', type=str)
    parser.add_argument('--backbone', default='crossPred', action='store',choices = ['crossPred', 'DRAEM'])
    parser.add_argument('--class_name', default='all', action='store')
    parser.add_argument('--loss_mode', default='MSE', action='store', choices = ['MSE', 'Cos', 'MSE_Cos'])
    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        DRAEM_train(args)
