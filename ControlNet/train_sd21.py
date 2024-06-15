from share import *
import os
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from datasets.processed_ACDC import RandomGenerator, ACDCdataset
from datasets import dataset_synapse # Synapse_dataset, RandomGenerator
from datasets import dataset_acdc # BaseDataSets, RandomGenerator
from datasets import dataset_polyp
from datasets import fundus_dataloader
# from datasets import prostate_dataloader
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
    train_path = '/scratch/bbmr/ymp5078/segmentations/data/processed_ACDC'
    img_size = 512
    batch_size = 16
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    
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
        'fundus': {
            'root_path': '/scratch/bbmr/ymp5078/segmentations/data/Domain_gen_data/fundus/',
            'list_dir': None,
            'num_classes': 3,
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
        
    }[args.dataset.split('_')[0]]
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
    elif 'fundus' in args.dataset:
        splitid = int(args.dataset.split('_')[1])
        dataset = fundus_dataloader.FundusSegmentation(base_dir=dataset_config['root_path'],transform=fundus_dataloader.composed_transforms_test,splitid=[splitid])
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
