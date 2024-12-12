import random
import numpy as np
import os
from skimage.io import imread

import torch
from torch.utils import data
import torchvision.transforms.functional as TF


class SegDataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        gen_input_path: str = None,
        use_aug: bool = False,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.gen_input_path = gen_input_path
        self.use_aug = use_aug
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]

        x, y = imread(input_ID), imread(target_ID)

        gen_x = None
        if self.use_aug:
            gen_x = self.transform_input(x)
        elif self.gen_input_path is not None: # load generated image
            gen_input_ID = os.path.join(self.gen_input_path,os.path.basename(input_ID).split('.')[0]+'.npz')
            with np.load(gen_input_ID) as npz_data:
                gen_image = npz_data['image']
            gen_x = gen_image[np.random.choice(len(gen_image))]
            
            gen_x = self.transform_input(gen_x)


        x = self.transform_input(x)
        y = self.transform_target(y)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)
                if gen_x is not None:
                    gen_x = TF.hflip(gen_x)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)
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
            if gen_x is not None:
                gen_x = TF.affine(gen_x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
        if gen_x is not None:
            return x.float(), y.float(), gen_x.float()

        return x.float(), y.float()

