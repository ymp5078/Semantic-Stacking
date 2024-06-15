import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from utils.visualize import get_color_pallete, vocpallete


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # if random.random() > 0.5:
        #     image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(-1).expand((-1,-1,3))
        # print(image.max(),image.min())
        gt_seg = get_color_pallete(label).convert('RGB')

        # to [0,1]
        gt_seg = torch.from_numpy(np.array(gt_seg).astype(np.float32))# / 255.0)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label.long(),'seg_image':gt_seg}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.class_names = ['background', 'Aorta', 'Gallbladder', 'Left Kidney', 'Right Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach']

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        
        unique_classes = np.unique(sample['label'])
        prompt = "A 2D slice of a abdomen CT scan showing " + ", ".join([self.class_names[c] for c in unique_classes])
        # print(sample['image'].shape,sample['seg_image'].shape)
        return dict(jpg=sample['image'], txt=prompt, hint=sample['seg_image'], path=sample['case_name'])