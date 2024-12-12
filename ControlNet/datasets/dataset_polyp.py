import random
from skimage.io import imread
from scipy.ndimage.interpolation import zoom
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms
import glob
from utils.visualize import get_color_pallete, vocpallete


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)  # why not 3?
            if len(label.shape) > 2:
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y,1), order=0)
                label = (label.sum(-1) > 2).astype(int)
            else:
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.)
        gt_seg = get_color_pallete(label).convert('RGB')

        # to [0,1]
        gt_seg = torch.from_numpy(np.array(gt_seg).astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long(),'seg_image':gt_seg}
        return sample
    
class SegDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        dataset: str,
        transform=None,
    ):
        if dataset == "Kvasir":
            img_path = root + "images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = root + "masks/*"
            target_paths = sorted(glob.glob(depth_path))
        elif dataset == "CVC":
            img_path = root + "Original/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = root + "Ground Truth/*"
            target_paths = sorted(glob.glob(depth_path))
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform = transform
        self.class_names = ['background', 'polyps']


    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]

        
        image, label = imread(input_ID), (imread(target_ID) > 127.5).astype(int)
        sample = {'image':image,'label':label,}
        if self.transform is not None:
            sample = self.transform(sample)
        unique_classes = np.unique(sample['label'])
        
        prompt = 'A image of the human gastrointestinal tract captured by colonoscope showing ' + ", ".join([self.class_names[c] for c in unique_classes])
        sample['case_name'] = input_ID.split('/')[-1].split('.')[0]
        return dict(jpg=sample['image'], txt=prompt, hint=sample['seg_image'], path=sample['case_name'])

