CUDA_VISIBLE_DEVICES=0 python train_acdc.py --dataset ACDC --vit_name R50-ViT-B_16 --exp TU_consist_ACDC --gen_image  --consist_loss_weight 1.0 --kd_loss_weight 1.0
python test_acdc.py --dataset ACDC --vit_name R50-ViT-B_16 --exp TU_consist_ACDC

CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16 --exp TU_consist_Synapse --gen_image  --consist_loss_weight 1.0 --kd_loss_weight 1.0
python test.py --dataset Synapse --vit_name R50-ViT-B_16 --exp TU_consist_Synapse