import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label, gen_image=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    if gen_image is not None:
        gen_image = np.rot90(gen_image, k)
        gen_image = np.flip(gen_image, axis=axis).copy()

    return image, label, gen_image


def random_rotate(image, label, gen_image=None):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    if gen_image is not None:
        gen_image = ndimage.rotate(gen_image, angle, order=0, reshape=False)
    return image, label, gen_image


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        gen_image = sample.get('gen_image',None)

        if random.random() > 0.5:
            image, label, gen_image = random_rot_flip(image, label, gen_image)
        elif random.random() > 0.5:
            image, label, gen_image = random_rotate(image, label, gen_image)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if gen_image is not None and (gen_image.shape[0]!= self.output_size[0] or gen_image.shape[1] != self.output_size[1]):
            gen_image = zoom(gen_image, (self.output_size[0] / gen_image.shape[0], self.output_size[1] / gen_image.shape[1]), order=0)  # the default is 0
            gen_image = torch.from_numpy(gen_image.astype(np.float32)).unsqueeze(0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'gen_image':gen_image}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, gen_image_dir=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

        # custom, add generated images for the same seg map
        
        self.gen_image_dir = gen_image_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
            sample = {'image': image, 'label': label}
            if self.gen_image_dir is not None:
                case = self.sample_list[idx].strip('\n')
                gen_image_path = os.path.join(self.gen_image_dir,case+'.npz')
                # print(gen_image_path)
                with np.load(gen_image_path) as npz_data:
                    gen_image = npz_data['image']
                gen_image = gen_image[np.random.choice(len(gen_image))]
                sample['gen_image'] = gen_image.mean(-1)
                
            sample = self.transform(sample)
            if sample['gen_image'] is None:
                del sample['gen_image']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

            sample = {'image': image, 'label': label}

            if self.transform:
                sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
