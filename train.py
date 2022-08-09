import torch
from data_loader import MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from models.model_DRAEM import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
from torchvision import transforms
import tqdm

import torch.nn.functional as F
import random

from dataloader_zzx import MVTecDataset
from evaluation import evaluation

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def mean(list_x):
    return sum(list_x)/len(list_x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
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

        
        
def add_Gaussian_noise(x, noise_res, noise_std, img_size):
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

    ns = F.upsample_bilinear(ns, size=[img_size, img_size])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(128))
    roll_y = random.choice(range(128))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

    mask = x.sum(dim=1, keepdim=True) > 0.01
    ns *= mask # Only apply the noise in the foreground.
    res = x + ns
    
    return res
        
        
def DRAEM_train(args, dataloader, model, model_seg, loss_l2, loss_ssim, loss_focal, optimizer, visualizer, scheduler, run_name):
    n_iter = 0
    for epoch in range(args.epochs):
        print("Epoch: "+str(epoch))
        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            aug_gray_batch = sample_batched["augmented_image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()

            gray_rec = model(aug_gray_batch)
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = loss_l2(gray_rec,gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)

            segment_loss = loss_focal(out_mask_sm, anomaly_mask)
            loss = l2_loss + ssim_loss + segment_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if args.visualize and n_iter % 20 == 0:
                visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
            if args.visualize and n_iter % 50 == 0:
                t_mask = out_mask_sm[:, 1:, :, :]
                visualizer.visualize_image_batch(aug_gray_batch, n_iter, image_name='batch_augmented')
                visualizer.visualize_image_batch(gray_batch, n_iter, image_name='batch_recon_target')
                visualizer.visualize_image_batch(gray_rec, n_iter, image_name='batch_recon_out')
                visualizer.visualize_image_batch(anomaly_mask, n_iter, image_name='mask_target')
                visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')


            n_iter +=1

        scheduler.step()

        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
        torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))
        
        
# def autoencoder_train()


def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+"Guassian_blur"
    run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_" + args.model + "_" + args.process_method
    ckp_path = os.path.join('/home/zhaoxiang/baselines/pretrain/output', run_name, 'last.pth')
    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))


    if args.backbone == 'DRAEM':
        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.cuda()
        model.apply(weights_init)

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.cuda()
        model_seg.apply(weights_init)
        
        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM()
        loss_focal = FocalLoss()

        dataset = MVTecDRAEMTrainDataset('/home/zhaoxiang/dataset/{}/train/good'.format(args.dataset_name), args.anomaly_source_path, resize_shape=[args.img_size, args.img_size])
        dataloader = DataLoader(dataset, batch_size=args.bs,
                            shuffle=True, num_workers=16)

        optimizer = torch.optim.Adam([
                                        {"params": model.parameters(), "lr": args.lr},
                                        {"params": model_seg.parameters(), "lr": args.lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)


   


        DRAEM_train(args, dataloader, model, model_seg, loss_l2, loss_ssim, loss_focal, optimizer, visualizer, scheduler, run_name)
        
    elif args.backbone == 'noise':
        from model_noise import UNet
        main_path = '/home/zhaoxiang/dataset/{}'.format(args.dataset_name)
    
    # data preparation
        data_transform, gt_transform = get_data_transforms(args.img_size, args.img_size)
        # data_transform, gt_transform = cutpaste_transform(args.img_size, args.img_size)
        test_transform, _ = get_data_transforms(args.img_size, args.img_size)
    
        dirs = os.listdir(main_path)
        
        for dir_name in dirs:
            if 'train' in dir_name:
                train_dir = dir_name
            elif 'test' in dir_name:
                if 'label' in dir_name:
                    label_dir = dir_name
                else:
                    test_dir = dir_name
                
        dirs = [train_dir, test_dir, label_dir]                
        
        device = torch.device('cuda:1')
        n_input = 1
        n_classes = 1           # the target is the reconstructed image
        depth = 4
        wf = 6
        
        if args.model == 'ws_skip_connection':
            model = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).to(device)
        elif args.model == 'DRAEM_reconstruction':
            model = ReconstructiveSubNetwork(in_channels=n_input, out_channels=n_input).to(device)
        elif args.model == 'DRAEM_discriminitive':  
            model = DiscriminativeSubNetwork(in_channels=n_input, out_channels=n_input).to(device)
        
        train_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='train', dirs = dirs, data_source=args.experiment_name)
        val_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name)
        test_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.bs, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = args.bs, shuffle = False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
         
         
        loss_MSE = torch.nn.MSELoss()
        
        loss_l1 = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # if (epoch) % 10 == 0:
        # epoch = 0
        # model.eval()
        # evaluation(args, model, test_dataloader, epoch, device, loss_l1, visualizer, run_name)
        # for epoch in tqdm(range(args.epochs)):
        for epoch in range(args.epochs):
            model.train()
            loss_list = []
            for img in train_dataloader:         

                img = img.to(device)                
                
                input = add_Gaussian_noise(img, args.noise_res, args.noise_std, args.img_size)         # if noise -> reconstruction
                
                output = model(input)
                # loss = loss_MSE(input, output)
                loss = loss_l1(img, output)

                # loss back propogation
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                loss_list.append(loss.item())
                
            print('epoch [{}/{}], loss:{:.4f}'.format(args.epochs, epoch, mean(loss_list)))
            
            
            visualizer.plot_loss(mean(loss_list), epoch, loss_name='L1_loss')
            visualizer.visualize_image_batch(input, epoch, image_name='input')
            visualizer.visualize_image_batch(img, epoch, image_name='target')
            visualizer.visualize_image_batch(output, epoch, image_name='output')
            
            
            
            # print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, epochs, np.mean(loss_list)))
            # with open(result_path, 'a') as f:
            #     # f.writelines('epoch [{}/{}], loss:{:.4f}, \n'.format(epoch+1, epochs, np.mean(loss_list)))
            #     f.writelines('epoch [{}/{}], loss:{:.4f}, loss_reconstruction:{:.4f}, loss_feature:{:.4f} \n'.format(epoch+1, epochs, np.mean(loss_list), np.mean(loss2_list), np.mean(loss1_list)))
            
            if (epoch) % 3 == 0:
                model.eval()
                error_list = []
                for img, gt, label, img_path, saves in val_dataloader:
                    img = img.to(device)
                    input = img
                    output = model(input)
                    loss = loss_l1(input, output)
                    
                    error_list.append(loss.item())
                
                print('eval [{}/{}], loss:{:.4f}'.format(args.epochs, epoch, mean(error_list)))
                visualizer.plot_loss(mean(error_list), epoch, loss_name='L1_loss_eval')
                visualizer.visualize_image_batch(input, epoch, image_name='target_eval')
                visualizer.visualize_image_batch(output, epoch, image_name='output_eval')
                
            if (epoch) % 10 == 0:
                model.eval()
                evaluation(args, model, test_dataloader, epoch, device, loss_l1, visualizer, run_name)
            
                torch.save(model.state_dict(), ckp_path)
                
            #     dice_value, auroc_px, auroc_sp, aupro_px = evaluation(stu_enc, tea_enc, bn, test_dataloader, device, epoch, config)
            #     # dice_value = evaluation(stu_enc, tea_enc, bn, test_dataloader, device, epoch, config)
            #     print('dice_score is:', dice_value, '\n'
            #         'Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            #     # print('dice_score is:', dice_value, '\n')
                
            #     with open(result_path, 'a') as f:
            #         f.writelines('dice_value [{}/{}] :{:.4f}, \n Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(epoch+1, epochs, dice_value, auroc_px, auroc_sp, aupro_px))
                    # f.writelines('Threshold:{}, dice_value :{:.4f}, \n'.format(config['threshold'], dice_value))
                
        
        

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
        train_on_device(args)

