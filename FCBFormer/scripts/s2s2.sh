# please replace the values to the actual path
CVC_PATH=[cvc_dataset_path]
KVASIR_PATH=[kvasir_dataset_path]
SAVEPATH=[s2s2_result_path]

python -W ignore train.py --dataset=CVC_gen --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/CVC-ClinicDB/ --gen-input-path /scratch/bbmr/ymp5078/segmentations/ControlNet/CVC_gen_16 --consist_loss_weight 0.4 --kd_loss_weight 0.4 --gen_image --multi-gpu true --save-path $SAVEPATH
python -W ignore train.py --dataset=Kvasir_gen --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/Kvasir-SEG/ --gen-input-path /scratch/bbmr/ymp5078/segmentations/ControlNet/Kvasir_gen_16 --consist_loss_weight 0.4 --kd_loss_weight 0.4 --gen_image --multi-gpu true --save-path $SAVEPATH

python predict.py --train-dataset=CVC_gen --test-dataset=CVC --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/CVC-ClinicDB/ --save-path $SAVEPATH
python predict.py --train-dataset=Kvasir_gen --test-dataset=CVC --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/CVC-ClinicDB/ --save-path $SAVEPATH
python predict.py --train-dataset=CVC_gen --test-dataset=Kvasir --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/Kvasir-SEG/ --save-path $SAVEPATH
python predict.py --train-dataset=Kvasir_gen --test-dataset=Kvasir --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/Kvasir-SEG/ --save-path $SAVEPATH

python eval.py --train-dataset=CVC_gen --test-dataset=CVC --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/CVC-ClinicDB/ --save-path $SAVEPATH
python eval.py --train-dataset=Kvasir_gen --test-dataset=CVC --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/CVC-ClinicDB/ --save-path $SAVEPATH
python eval.py --train-dataset=CVC_gen --test-dataset=Kvasir --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/Kvasir-SEG/ --save-path $SAVEPATH
python eval.py --train-dataset=Kvasir_gen --test-dataset=Kvasir --data-root=/scratch/bbmr/ymp5078/segmentations/data/Polyp_data/Kvasir-SEG/ --save-path $SAVEPATH