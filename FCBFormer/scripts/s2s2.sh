# please replace the values to the actual path
CVC_PATH=[cvc_dataset_path]
KVASIR_PATH=[kvasir_dataset_path]
SAVEPATH=[s2s2_result_path]

python -W ignore train.py --dataset=CVC_gen --data-root=$CVC_PATH --gen-input-path [gen_output_path] --consist_loss_weight 0.4 --kd_loss_weight 0. --gen_image --multi-gpu true --save-path $SAVEPATH
python -W ignore train.py --dataset=Kvasir_gen --data-root=$KVASIR_PATH --gen-input-path [gen_output_path] --consist_loss_weight 0.4 --kd_loss_weight 0. --gen_image --multi-gpu true --save-path $SAVEPATH

python predict.py --train-dataset=CVC_gen --test-dataset=CVC --data-root=$CVC_PATH --save-path $SAVEPATH
python predict.py --train-dataset=Kvasir_gen --test-dataset=CVC --data-root=$CVC_PATH --save-path $SAVEPATH
python predict.py --train-dataset=CVC_gen --test-dataset=Kvasir --data-root=$KVASIR_PATH --save-path $SAVEPATH
python predict.py --train-dataset=Kvasir_gen --test-dataset=Kvasir --data-root=$KVASIR_PATH --save-path $SAVEPATH

python eval.py --train-dataset=CVC_gen --test-dataset=CVC --data-root=$CVC_PATH --save-path $SAVEPATH
python eval.py --train-dataset=Kvasir_gen --test-dataset=CVC --data-root=$CVC_PATH --save-path $SAVEPATH
python eval.py --train-dataset=CVC_gen --test-dataset=Kvasir --data-root=$KVASIR_PATH --save-path $SAVEPATH
python eval.py --train-dataset=Kvasir_gen --test-dataset=Kvasir --data-root=$KVASIR_PATH --save-path $SAVEPATH