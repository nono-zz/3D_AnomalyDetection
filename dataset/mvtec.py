import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils.mvtec3d_util import read_tiff_organized_pc, resize_organized_pc, organized_pc_to_depth_map
import numpy as np

CLASS_NAMES  = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'foam', 'peach', 'potato', 'rope', 'tire']

class MVTecDataset(Dataset):
    def __init__(self, dataset_path, class_name='bottle', is_train=True, resize=256, cropsize=224, wild_ver=False):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        self.rgb_paths, self.xyz_paths, self.y, self.mask = self.load_dataset_folder()

        if wild_ver:
            self.transform_x =   T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                            T.RandomRotation(10),
                                            T.RandomCrop(cropsize),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

            self.transform_mask =    T.Compose([T.Resize(resize, Image.NEAREST),
                                                T.RandomRotation(10),
                                                T.RandomCrop(cropsize),
                                                T.ToTensor()])
                                                
        else:
            self.transform_x =   T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                            T.CenterCrop(cropsize),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

            self.transform_mask =    T.Compose([T.Resize(resize, Image.NEAREST),
                                                T.CenterCrop(cropsize),
                                                T.ToTensor()])
            
            self.transform_FPFH = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                            T.CenterCrop(cropsize),
                                            T.ToTensor(),
                                            T.Normalize(mean=[0.456],
                                                        std=[0.224])])

    def __getitem__(self, idx):
        # x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        rgb_path, pc_path, label, mask = self.rgb_paths[idx], self.xyz_paths[idx], self.y[idx], self.mask[idx]
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_img = self.transform_x(rgb_img)
        
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        
        organized_pc = read_tiff_organized_pc(pc_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)
        return rgb_img, resized_organized_pc, resized_depth_map_3channel, label, mask
    

    def __len__(self):
        return len(self.rgb_paths)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        # x, y, mask = [], [], []
        rgb_tot_fpath_list, xyz_tot_fpath_list, y, mask = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        # gt_dir = os.path.join(self.dataset_path, self.class_name, 'gt')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            
            rgb_type_dir = os.path.join(img_type_dir, 'rgb')
            xyz_type_dir = os.path.join(img_type_dir, 'xyz')
            
            # read RGB
            rgb_fpath_list = sorted([os.path.join(rgb_type_dir, f)
                                    for f in os.listdir(rgb_type_dir)
                                    if f.endswith('.png')])
            
            rgb_tot_fpath_list.extend(rgb_fpath_list)
            
            # read xyz
            xyz_type_dir = sorted([os.path.join(xyz_type_dir, f)
                                    for f in os.listdir(xyz_type_dir)
                                    if f.endswith('.tiff')])
            
            xyz_tot_fpath_list.extend(xyz_type_dir)
            

            if img_type == 'good':
                y.extend([0] * len(rgb_fpath_list))
                mask.extend([None] * len(rgb_fpath_list))
            else:
                gt_type_dir = os.path.join(img_dir, img_type, 'gt')
                y.extend([1] * len(rgb_fpath_list))
                # gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in rgb_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(rgb_tot_fpath_list) == len(y), 'number of x and y should be same'
        assert len(rgb_tot_fpath_list) == len(xyz_tot_fpath_list), 'length of rgb and xyz is not the same'

        # return list(x), list(y), list(mask)
        return list(rgb_tot_fpath_list), list(xyz_tot_fpath_list), list(y), list(mask)
