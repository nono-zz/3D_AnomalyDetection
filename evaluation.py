import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from numpy import ndarray
import pandas as pd
from sklearn.metrics import auc
from skimage import measure

# from torch.utils.tensorboard import SummaryWriter
from utils.tensorboard_visualizer import TensorboardVisualizer

from sklearn.metrics import roc_auc_score

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def mean(list_x):
    return sum(list_x)/len(list_x)


def cal_anomaly_map(rgb, depth, rgbPred, depthPred):
    out_size = rgb.shape[-1]
    # anomaly_map = np.zeros([out_size, out_size])

    a_map_rgb = 1 - F.cosine_similarity(rgb, rgbPred)
    a_map_depth = 1 - F.cosine_similarity(depth, depthPred)
    
    a_map_rgb = a_map_rgb[0, :, :].to('cpu').detach().numpy()
    a_map_depth = a_map_depth[0, :, :].to('cpu').detach().numpy()
    
    a_map = a_map_rgb + a_map_depth
    
    return a_map, a_map_rgb, a_map_depth
   
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

        

def evaluation(args, model, test_loader, class_name, visualizer, device, epoch):
    model.eval()


    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        # for img, pc, gt, label, _ in dataloader:
        for (rgb, depth, mask, label) in test_loader:
            rgb = rgb.to(device)
            depth = depth.to(device)
            
            rgbPred, depthPred = model(rgb, depth)
            

            a_map, a_map_rgb, a_map_depth = cal_anomaly_map(rgb, depth, rgbPred, depthPred)
            # anomaly_map = gaussian_filter(anomaly_map, sigma=4)         # why use gaussian filter to blur the anomaly map?
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            
            if label.item()!=0:
                aupro_list.append(compute_pro(mask.squeeze(0).cpu().numpy().astype(int),
                                              a_map[np.newaxis,:,:]))
            gt_list_px.extend(mask.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(a_map.ravel())
            gt_list_sp.append(np.max(mask.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(a_map))


        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)



def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    
    """
    Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df_new = pd.DataFrame([[mean(pros), fpr, th]],  columns=['pro', 'fpr', 'threshold'])
        # df = df.concat({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df = pd.concat([df, df_new], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
