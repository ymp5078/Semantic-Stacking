CUDA_VISIBLE_DEVICES=0 python train_acdc.py --dataset ACDC --vit_name R50-ViT-B_16
python test_acdc.py --dataset ACDC --vit_name R50-ViT-B_16

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
python test.py --dataset Synapse --vit_name R50-ViT-B_16