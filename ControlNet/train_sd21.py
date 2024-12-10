from share import *
import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from datasets import dataset_synapse # Synapse_dataset, RandomGenerator
from datasets import dataset_acdc # BaseDataSets, RandomGenerator
from datasets import dataset_polyp
from datasets import CardiacDataset
from datasets import AbdominalDataset
from torchvision import transforms




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate fake data for the given dataset.')
    parser.add_argument('--dataset', default='acdc')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--out', '-o', type=str, default='result')
    args = parser.parse_args()
    
    
    # Configs
    resume_path = './models/control_sd21_ini.ckpt'
    img_size = 512
    batch_size = 16
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # Misc
    dataset_config = {
        'acdc': {
            'root_path': '[data_dir]',
            'list_dir': None,
            'num_classes': 4,
        },
        'Synapse': {
            'root_path': '[data_dir]',
            'list_dir': '[data_dir]/lists_Synapse',
            'num_classes': 9,
        },
        'Kvasir': {
            'root_path': '[data_dir]',
            'list_dir': None,
            'num_classes': 2,
        },
        'CVC': {
            'root_path': '[data_dir]',
            'list_dir': None,
            'num_classes': 2,
        },
        'LGE': {
            'root_path': '[data_dir]',
            'list_dir': None,
            'num_classes': 4,
        },
        'bSSFP': {
            'root_path': '[data_dir]',
            'list_dir': None,
            'num_classes': 4,
        },
        
        'SABSCT': {
            'root_path': '[data_dir]',
            'list_dir': None,
            'num_classes': 5,
        },
        'CHAOST2': {
            'root_path': '[data_dir]',
            'list_dir': None,
            'num_classes': 5,
        },
        
    }[args.dataset.split('_')[0]]

    

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

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
        # splitid = int(args.dataset.split('_')[1])
        dataset = CardiacDataset.get_gen_training(modality=['LGE'], location_scale=None,  tile_z_dim = 3)
    elif 'bSSFP' in args.dataset:
        # splitid = int(args.dataset.split('_')[1])
        dataset = CardiacDataset.get_gen_training(modality=['bSSFP'], location_scale=None,  tile_z_dim = 3)
    elif 'SABSCT' in args.dataset:
        dataset = AbdominalDataset.get_gen_training(modality=['SABSCT'], location_scale=None)
    elif 'CHAOST2' in args.dataset:
        dataset = AbdominalDataset.get_gen_training(modality=['CHAOST2'], location_scale=None)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger],max_epochs=args.max_epochs,default_root_dir=f'./result_logs/{args.dataset}')


    # Train!
    trainer.fit(model, dataloader)
