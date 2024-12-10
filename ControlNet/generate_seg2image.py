from share import *
import config
import os
from tqdm import tqdm

import cv2
import einops
import numpy as np
import torch
import random
import argparse
from PIL import Image
from torchvision import transforms

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from datasets.processed_ACDC import RandomGenerator, ACDCdataset
from datasets import dataset_synapse # Synapse_dataset, RandomGenerator
from datasets import dataset_acdc # BaseDataSets, RandomGenerator
from datasets import dataset_polyp
from datasets import CardiacDataset
from datasets import AbdominalDataset




def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        # input_image = HWC3(input_image)
        # detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        detected_map = input_image
        # img = resize_image(input_image, image_resolution)
        H, W, C = detected_map.shape


        control = detected_map.float().cuda() #/ 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        if a_prompt is not None:
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        else:
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}

        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        vis_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        x_samples = einops.rearrange(x_samples, 'b c h w -> b h w c').cpu().numpy()

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results, vis_samples

# Misc
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate fake data for the given dataset.')
    parser.add_argument('--dataset', default='acdc')
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--train_path', type=str, default='/scratch/bbmr/ymp5078/segmentations/data/processed_ACDC')
    parser.add_argument('--model_path', type=str, default='/scratch/bbmr/ymp5078/segmentations/ControlNet/lightning_logs/version_2906611/checkpoints/test_model.ckpt')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--seed', '-s', type=int, default=-1)
    args = parser.parse_args()
    
    img_size = 512
    
    apply_uniformer = UniformerDetector()

    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(args.model_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    dataset_config = {
        'acdc': {
            # 'Dataset': ACDC_dataset,  # datasets.dataset_acdc.BaseDataSets,
            'root_path': '/scratch/bbmr/ymp5078/segmentations/data/ACDC',
            'list_dir': None,
            'num_classes': 4,
        },
        'Synapse': {
            'root_path': '/scratch/bbmr/ymp5078/segmentations/data/Synapse/train_npz',
            'list_dir': '/scratch/bbmr/ymp5078/segmentations/data/Synapse/lists_Synapse',
            'num_classes': 9,
        },
        'Kvasir': {
            'root_path': '/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/Kvasir-SEG/',
            'list_dir': None,
            'num_classes': 2,
        },
        'CVC': {
            'root_path': '/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/CVC-ClinicDB/',
            'list_dir': None,
            'num_classes': 2,
        },
        'LGE': {
            'root_path': '',
            'list_dir': None,
            'num_classes': 4,
        },
        'bSSFP': {
            'root_path': '',
            'list_dir': None,
            'num_classes': 4,
        },
        
        'SABSCT': {
            'root_path': '',
            'list_dir': None,
            'num_classes': 5,
        },
        'CHAOST2': {
            'root_path': '',
            'list_dir': None,
            'num_classes': 5,
        },
        

    }[args.dataset.split("_")[0]]
    if args.dataset == 'acdc':
        dataset = dataset_acdc.BaseDataSets(base_dir=dataset_config['root_path'], split="train", transform=transforms.Compose([
                        dataset_acdc.RandomGenerator([img_size,img_size])]))
    elif args.dataset == 'Synapse':
        dataset = dataset_synapse.Synapse_dataset(base_dir=dataset_config['root_path'], list_dir=dataset_config['list_dir'], split="train",
                               transform=transforms.Compose(
                                   [dataset_synapse.RandomGenerator(output_size=[img_size,img_size])]))
    elif args.dataset == 'Kvasir':
        dataset = dataset_polyp.SegDataset(dataset=args.dataset, root=dataset_config['root_path'], transform=dataset_polyp.RandomGenerator(output_size=[img_size,img_size]))
    elif args.dataset == 'CVC':
        dataset = dataset_polyp.SegDataset(dataset=args.dataset, root=dataset_config['root_path'], transform=dataset_polyp.RandomGenerator(output_size=[img_size,img_size]))
    elif 'LGE' in args.dataset:
        img_size = 192
        # splitid = int(args.dataset.split('_')[1])
        dataset = CardiacDataset.get_gen_training(modality=['LGE'], location_scale=None,  tile_z_dim = 3)
    elif 'bSSFP' in args.dataset:
        img_size = 192
        # splitid = int(args.dataset.split('_')[1])
        dataset = CardiacDataset.get_gen_training(modality=['bSSFP'], location_scale=None,  tile_z_dim = 3)
    elif 'SABSCT' in args.dataset:
        img_size = 192
        dataset = AbdominalDataset.get_gen_training(modality=['SABSCT'], location_scale=None)
    elif 'CHAOST2' in args.dataset:
        img_size = 192
        dataset = AbdominalDataset.get_gen_training(modality=['CHAOST2'], location_scale=None)
    RGB_datasets = ['CVC','Kvasir','fundus_1','fundus_2','fundus_3','fundus_4']

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        out_dir = args.out
        if os.path.exists(os.path.join(out_dir,f"{sample['path']}.npz")): continue
        results, vis_result = process(input_image=sample['hint'], 
                          prompt=sample['txt'], 
                          a_prompt=None, 
                          n_prompt="", 
                          num_samples=args.num_samples, 
                          image_resolution=img_size, 
                          detect_resolution=img_size, 
                          ddim_steps=50, 
                          guess_mode=False, 
                          strength=1.0, 
                          scale=9.0, 
                          seed=args.seed, 
                          eta=0.0)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir,f"{sample['path']}.npz")

        # This depends on the original dataloader or the given dataset
        if args.dataset in RGB_datasets:
            np.savez(save_path, image=vis_result[1:])
        else:
            np.savez(save_path, image=results[1:])
        vis_out_dir = os.path.join(out_dir,'vis')
        os.makedirs(vis_out_dir, exist_ok=True)
        for i,result in tqdm(enumerate(results[1:])):
            # print(result.shape)
            save_path = os.path.join(vis_out_dir,f"{sample['path']}_{i}.jpg")
            im = Image.fromarray(vis_result[i])
            im.save(save_path)
    print('done!')