import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
from data.base_dataset import BaseDataset
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt


def my_segmentation_transforms(image, another, segmentation):
    segmentation = segmentation.unsqueeze(0)
    
    # if random.random() > 0.75:
    #     w, h = TF.get_image_size(image)
    #     top = random.randint(int(0.4*h),int(0.5*h))
    #     left = random.randint(int(0.2*w),int(0.25*w))
    #     k = random.uniform(0.5,0.6)
    #     width = int(k*w)
    #     height = int(k*h)

    #     image = TF.crop(image, top=top, left=left, height=height, width=width)
    #     another = TF.crop(another, top=top, left=left, height=height, width=width)
    #     segmentation = TF.crop(segmentation, top=top, left=left, height=height, width=width)
    #     image = TF.resize(image, size=[h,w])
    #     another = TF.resize(another, size=[h,w])
    #     segmentation = TF.resize(segmentation, size=[h,w])
        
    if random.random() > 0.667:
        contrast = random.uniform(0.5,2)
        image = TF.adjust_contrast(image, contrast)
        
    if random.random() > 0.667:
        brightness = random.uniform(0.5,2)
        image = TF.adjust_brightness(image, brightness)
        
    # if random.random() > 0.5:
    #     image = TF.hflip(image)
    #     another = TF.hflip(another)
    #     segmentation = TF.hflip(segmentation)

    # if random.random() > 0.667:
    #     angle = random.randint(-30, 30)
    #     image = TF.rotate(image, angle, fill=0)
    #     another = TF.rotate(another, angle, fill=0)
    #     segmentation = TF.rotate(segmentation, angle, fill=0)

    return image, another, segmentation.squeeze(0)
class kittiCalibInfo():
    """
    Read calibration files in the kitti dataset,
    we need to use the intrinsic parameter of the cam2
    """
    def __init__(self, filepath):
        """
        Args:
            filepath ([str]): calibration file path (AAA.txt)
        """
        self.data = self._load_calib(filepath)

    def get_cam_param(self):
        """
        Returns:
            [numpy.array]: intrinsic parameter of the cam2
        """
        return self.data['P2']

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        P0 = np.reshape(rawdata['P0'], (3,4))
        P1 = np.reshape(rawdata['P1'], (3,4))
        P2 = np.reshape(rawdata['P2'], (3,4))
        P3 = np.reshape(rawdata['P3'], (3,4))
        R0_rect = np.reshape(rawdata['R0_rect'], (3,3))
        Tr_velo_to_cam = np.reshape(rawdata['Tr_velo_to_cam'], (3,4))

        data['P0'] = P0
        data['P1'] = P1
        data['P2'] = P2
        data['P3'] = P3
        data['R0_rect'] = R0_rect
        data['Tr_velo_to_cam'] = Tr_velo_to_cam

        return data

    def _read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data





class multivkittiDataset(BaseDataset):
    """dataloader for vkitti2 dataset"""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.use_sne = opt.use_sne
        
        self.use_size = (opt.useWidth, opt.useHeight)
        

        if opt.phase == "train":
            with open(os.path.join(opt.split_scheme, 'train.txt'), 'r') as f:
                self.name_list = f.readlines()
        elif opt.phase == "val":
            with open(os.path.join(opt.split_scheme, 'val.txt'), 'r') as f:
                self.name_list = f.readlines()
        else:
            with open(os.path.join(opt.split_scheme, 'test.txt'), 'r') as f:
                self.name_list = f.readlines()

        self.color_list = [(210, 0, 200), (90, 200, 255), (0, 199, 0), (90, 240, 0), (140, 140, 140), 
                           (100, 60, 100), (250, 100, 255), (255, 255, 0), (200, 200, 0), (255, 130, 0), 
                           (80, 80, 80), (160, 60, 60), (255, 127, 80), (0, 139, 139), (0, 0, 0)]

        self.num_labels = len(self.color_list)
        
    def __getitem__(self, index):
        
        rgb_path, depth_path, seg_path, normal_path = self.name_list[index].replace('\n', '').split(', ')

        rgb_image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        oriHeight, oriWidth, _ = rgb_image.shape
        if self.opt.phase == 'test' and self.opt.no_label:
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        else:
            label_image = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            for i, color in enumerate(self.color_list):
                label_i = cv2.inRange(label_image, color, color)
                label_i[label_i > 0] = 1
                label += label_i * i
            
        # resize image to enable sizes divide 32
        rgb_image = cv2.resize(rgb_image, self.use_size)
        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)

        # another_image will be normal when using SNE, otherwise will be depth
        if self.use_sne:
            if self.opt.sne == 'd2nt':
                camParam = np.array([[725.0087, 0, 620.5, 0],
                                    [0, 725.0087, 187, 0],
                                    [0, 725.0087, 0, 0]])
                depth = torch.tensor(depth_image.astype(np.float32))
                normal = self.sne_model(depth, camParam)
                another_image = normal.cpu().numpy()
                another_image = np.transpose(another_image, [1, 2, 0])
                another_image = cv2.resize(another_image, self.use_size)
            else:
                another_image = rgb_image
        else:
            # if self.opt.disp == 'depth':
            another_image = depth_image.astype(np.float32) / depth_image.max()
            # depth_image = depth_image.astype(np.float32) 
            # disp = np.ones_like(depth_image) / depth_image
            # disp[np.isnan(disp)] = 0
            # disp[np.isinf(disp)] = 0
            # another_image = disp / disp.max()
            another_image = cv2.resize(another_image, self.use_size)
            another_image = another_image[:,:,np.newaxis]

        rgb_image = rgb_image.astype(np.float32) / 255
        rgb_image = transforms.ToTensor()(rgb_image)
        another_image = transforms.ToTensor()(another_image)

        label = torch.from_numpy(label)

        # if self.opt.phase == 'train':
        #     rgb_image, another_image, label = my_segmentation_transforms(rgb_image, another_image, label)



        label = label.type(torch.LongTensor)


        return {'rgb_image': rgb_image, 'another_image': another_image, 'label': label,
                'path': self.name_list[index], 'oriSize': (oriWidth, oriHeight)}

    def __len__(self):
        return len(self.name_list)

    def name(self):
        return 'kitti'