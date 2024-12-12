import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage import io
import cv2
class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', list_dir=None, transform=None, gen_image_dir=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        train_ids, val_ids, test_ids = self._get_ids()
        if self.split.find('train') != -1:
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split.find('val') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        elif self.split.find('test') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        print("total {} samples".format(len(self.sample_list)))

        # custom, add generated images for the same seg map
        
        self.gen_image_dir = gen_image_dir

    def _get_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
        training_set = [i for i in all_cases_set if i not in testing_set+validation_set]

        return [training_set, validation_set, testing_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/ACDC_training_slices/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]  # fix sup_type to label
            sample = {'image': image, 'label': label}
            if self.gen_image_dir is not None: # load generated image
                gen_image_path = os.path.join(self.gen_image_dir,case.replace('.h5','.npz'))
                # print(gen_image_path)
                with np.load(gen_image_path) as npz_data:
                    gen_image = npz_data['image']
                gen_image = gen_image[np.random.choice(len(gen_image))]
                sample['gen_image'] = gen_image.mean(-1)
                
            sample = self.transform(sample)
            if sample['gen_image'] is None:
                del sample['gen_image']
        else:
            h5f = h5py.File(self._base_dir + "/ACDC_training_volumes/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = idx
        sample['case_name'] = case.replace('.h5', '')
        return sample


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
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # the default is 0
            label = zoom( label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if gen_image is not None and (gen_image.shape[0]!= self.output_size[0] or gen_image.shape[1] != self.output_size[1]):
            gen_image = zoom(gen_image, (self.output_size[0] / gen_image.shape[0], self.output_size[1] / gen_image.shape[1]), order=0)  # the default is 0
            gen_image = torch.from_numpy(gen_image.astype(np.float32)).unsqueeze(0)
        assert (image.shape[0] == self.output_size[0]) and (image.shape[1] == self.output_size[1])
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label, 'gen_image':gen_image}
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
