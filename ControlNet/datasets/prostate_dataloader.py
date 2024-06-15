from __future__ import print_function, division
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from glob import glob
import random
import copy
from skimage.transform import resize

class Normalize_tf(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        __mask = np.array(sample['label']).astype(np.uint8)
        img /= 127.5
        img -= 1.0
        _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
        _mask[__mask > 200] = 255
        # index = np.where(__mask > 50 and __mask < 201)
        _mask[(__mask > 50) & (__mask < 201)] = 128
        _mask[(__mask > 50) & (__mask < 201)] = 128

        __mask[_mask == 0] = 2
        __mask[_mask == 255] = 0
        __mask[_mask == 128] = 1

        # mask = to_multilabel(__mask) # dont use multilabel for generation
        sample['image'] = img
        sample['label'] = __mask
        gt_seg = get_color_pallete(sample['label']).convert('RGB')
        sample['gt_seg'] = gt_seg
        return sample

class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']

        assert img.width == mask.width
        assert img.height == mask.height
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'img_name': name}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32)#.transpose((2, 0, 1))
        gt_seg = np.array(sample['gt_seg']).astype(np.float32)#.transpose((2, 0, 1))
        map = np.array(sample['label']).astype(np.uint8)#.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        map = torch.from_numpy(map)#.float()
        gt_seg = torch.from_numpy(gt_seg)
        sample['image']=img
        sample['label']=map
        sample['gt_seg']=gt_seg
        return sample
    
composed_transforms_test = transforms.Compose([
        FixedResize((512,512)),
        Normalize_tf(),
        ToTensor()
    ])


class ProstateSegmentation(Dataset):
    """
    prostate segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('prostate'),
                 phase='train',
                 splitid=[2, 3, 4],
                 transform=None,
                 state='train',
                 gen_image_dir=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.image_pool = {'Domain1':[], 'Domain2':[], 'Domain3':[], 'Domain4':[], 'Domain5':[], 'Domain6':[]}
        self.label_pool = {'Domain1':[], 'Domain2':[], 'Domain3':[], 'Domain4':[], 'Domain5':[], 'Domain6':[]}
        self.img_name_pool = {'Domain1':[], 'Domain2':[], 'Domain3':[], 'Domain4':[], 'Domain5':[], 'Domain6':[]}

        # self.flags_DGS = ['gd', 'nd']
        # self.flags_REF = ['g', 'n']
        # self.flags_RIM = ['G', 'N', 'S']
        # self.flags_REF_val = ['V']
        self.splitid = splitid
        SEED = 1212
        random.seed(SEED)
        for id in splitid:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(id), 'image/')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.npy')
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path})
        self.transform = transform
        self._read_img_into_memory()
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break

        self.gen_image_dir = gen_image_dir
        
        self.class_names = ['background', 'prostate region']
        # Display stats
        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                if self.gen_image_dir is not None:
                    gen_image_path = os.path.join(self.gen_image_dir,_img_name.split('.')[0]+'.npz')
                    with np.load(gen_image_path) as npz_data:
                        gen_image = npz_data['image']
                    gen_image = gen_image[np.random.choice(len(gen_image))]
                    anco_sample['gen_image'] = Image.fromarray(gen_image)
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                # print(anco_sample)
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample=anco_sample
        unique_classes = np.unique(sample['label'].numpy())
        # print(unique_classes)
        prompt = "A e T2-weighted MRI image showing " + ", ".join([self.class_names[c] for c in unique_classes])
        
        return dict(jpg=sample['image'], txt=prompt, hint=sample['gt_seg'], path=sample['img_name'].split('.')[0])


    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            Flag = self.image_list[index]['image'].split('/')[-3]
            # print('img_flag',Flag)
            self.image_pool[Flag].append(
                Image.fromarray((resize(np.load(self.image_list[index]['image']),(256, 256), anti_aliasing=False, preserve_range=True)*127.5+127.5).astype(np.uint8)))
            # self.image_pool[Flag].append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = np.load(self.image_list[index]['label'])

            if self.state != 'prediction':
                _target = Image.fromarray(resize(_target, (256, 256), order=0, preserve_range=True).round().astype(np.uint8))
            # print(np.unique(_target))
            # print(_target.size)
            # print(_target.mode)
            self.label_pool[Flag].append(_target)
            # if self.split[0:4] in 'test':
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name)



    def __str__(self):
        return 'prostate(phase=' + self.phase+str(args.datasetTest[0]) + ')'


if __name__ == '__main__':
    import custom_transforms as tr
    from utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    voc_train = ProstateSegmentation(split='train1',
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = tmp
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

            break
    plt.show(block=True)
