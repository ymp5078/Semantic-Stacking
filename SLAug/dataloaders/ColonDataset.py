import numpy as np
import random
import multiprocessing

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
import glob

import os
from skimage.io import imread
from skimage.transform import resize

import torch
import torchvision.transforms.functional as TF
from .location_scale_augmentation import LocationScaleAugmentation

BASEDIR = "/scratch/bbmr/ymp5078/segmentations/data/Polyp_data"
GEN_BASEDIR = '/scratch/bbmr/ymp5078/segmentations/ControlNet'
LABEL_NAME = ['bg','fg']
class SegDataset(data.Dataset):
    def __init__(
        self,
        dataset: list,
        split:str,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
        location_scale=False
    ):
        dataset = dataset[0]
        if dataset == "Kvasir":
            img_path = BASEDIR + "/Kvasir-SEG/images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = BASEDIR + "/Kvasir-SEG/masks/*"
            target_paths = sorted(glob.glob(depth_path))
        elif dataset == "CVC":
            img_path = BASEDIR + "/CVC-ClinicDB/Original/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = BASEDIR + "/CVC-ClinicDB/Ground Truth/*"
            target_paths = sorted(glob.glob(depth_path))
        else:
            raise NotImplementedError
        
        self.all_label_names = LABEL_NAME
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.gen_input_path = os.path.join(GEN_BASEDIR,'CVC_gen_16' if dataset == "CVC" else 'Kvasir_gen_16') if split == 'train' else None
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine
        self.split = split

        if location_scale:
            print(f'Applying Location Scale Augmentation')
            self.location_scale = LocationScaleAugmentation(vrange=(0.,1.), background_threshold=0.01)
        else:
            self.location_scale = None
        
        self._set_subset(split=split)
        print(f'{split} split:',self.__len__())

    def __len__(self):
        return len(self.subset_map)
    
    def _set_subset(self,split):
        train_indices, test_indices, val_indices = split_ids(len(self.input_paths))
        if split in ['train','test']:
            self.subset_map = train_indices
        else:
            self.subset_map = val_indices

    def __getitem__(self, index: int):
        index = self.subset_map[index] # index from the subset
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]

        x, y = resize(imread(input_ID),output_shape=(192,192),anti_aliasing=False,order=1).astype(float), resize(imread(target_ID),output_shape=(192,192),anti_aliasing=False,order=0)
        
        gen_x = 1
        gen_image_GLA = 1
        gen_image_LLA = 1
        if self.gen_input_path is not None:
            gen_input_ID = os.path.join(self.gen_input_path,os.path.basename(input_ID).split('.')[0]+'.npz')
            with np.load(gen_input_ID) as npz_data:
                gen_image = npz_data['image']
            gen_x = gen_image[np.random.choice(len(gen_image))]
            gen_x = resize(gen_x,output_shape=(192,192),anti_aliasing=False,order=1)
            # if self.location_scale is not None:
            #     gen_image_GLA = self.location_scale.Global_Location_Scale_Augmentation(gen_x.copy())
            #     gen_image_LLA = self.location_scale.Local_Location_Scale_Augmentation(gen_x.copy(), y.astype(np.int32))
            #     gen_x = gen_image_GLA
            #     gen_image_LLA = self.transform_input(gen_image_LLA).float()
            # print('x',x.mean(),x.max(),x.min())
            # print('gen',gen_x.mean(),gen_x.max(),gen_x.min())
            gen_x = self.transform_input(gen_x).float()
        aug_img = 1
        if self.location_scale is not None:
            GLA = self.location_scale.Global_Location_Scale_Augmentation(x.copy())
            LLA = self.location_scale.Local_Location_Scale_Augmentation(x.copy(), y.astype(np.int32))
            x = GLA
            aug_img = LLA
            aug_img = self.transform_input(aug_img).float()
        x = self.transform_input(x).float()
        # y = np.expand_dims(y,axis=-1)
        # print(y.shape)
        y = torch.round(self.transform_target(y))
        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)
                if self.location_scale is not None:
                    aug_img  = TF.hflip(aug_img)
                    # if gen_x is not None:
                    #     gen_image_LLA = TF.hflip(gen_image_LLA)
                if gen_x is not None:
                    gen_x = TF.hflip(gen_x)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)
                if self.location_scale is not None:
                    aug_img  = TF.vflip(aug_img)
                    # if gen_x is not None:
                    #     gen_image_LLA = TF.vflip(gen_image_LLA)
                if gen_x is not None:
                    gen_x = TF.vflip(gen_x)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)
            if self.location_scale is not None:
                aug_img  = TF.affine(aug_img, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
                # if gen_x is not None:
                #     gen_image_LLA = TF.affine(gen_image_LLA, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            if gen_x is not None:
                gen_x = TF.affine(gen_x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
        
        # print(y.shape)
        sample = {"images": x,
                "labels":y[0].long(),
                "aug_images": aug_img,
                'gen_images': gen_x,
                'aug_gen_images':gen_image_LLA,
                }

        return sample

def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


def get_training(modality, location_scale,  tile_z_dim = 3, use_gen_image=False):
    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize((352, 352), antialias=True),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(),
        #   transforms.Resize((352, 352)),
            transforms.Grayscale()]
    )

    train_dataset = SegDataset(
        dataset=modality,
        split='train',
        transform_input=transform_input4train,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=True,
        location_scale=location_scale
    )

    # train_indices, test_indices, val_indices = split_ids(len(train_dataset.input_paths))

    # train_dataset = data.Subset(train_dataset, train_indices)

    return train_dataset

def get_validation(modality,  tile_z_dim = 3):

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), 
        #  transforms.Resize((352, 352)),
          transforms.Grayscale()]
    )

    val_dataset = SegDataset(
        dataset=modality,
        split='val',
        transform_input=transform_input4test,
        transform_target=transform_target,
    )


    return val_dataset

def get_test(modality,  tile_z_dim = 3):

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), 
        #  transforms.Resize((352, 352)), 
         transforms.Grayscale()]
    )

    train_dataset = SegDataset(
        dataset=modality,
        split='test',
        transform_input=transform_input4test,
        transform_target=transform_target,
    )


    return train_dataset



def get_datasets(input_paths, target_paths, batch_size, gen_input_path=None):

    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((352, 352)), transforms.Grayscale()]
    )

    train_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        gen_input_path=gen_input_path,
        transform_input=transform_input4train,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=True,
    )

    test_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    val_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    train_indices, test_indices, val_indices = split_ids(len(input_paths))

    train_dataset = data.Subset(train_dataset, train_indices)
    val_dataset = data.Subset(val_dataset, val_indices)
    test_dataset = data.Subset(test_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset



