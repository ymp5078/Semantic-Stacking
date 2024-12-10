# please replace the values to the actual path
CVC_PATH=[cvc_dataset_path]
KVASIR_PATH=[kvasir_dataset_path]
SAVEPATH=[baseline_result_path]

python -W ignore train.py --dataset=CVC --data-root=$CVC_PATH
python -W ignore train.py --dataset=Kvasir --data-root=$KVASIR_PATH

python predict.py --train-dataset=CVC --test-dataset=CVC --data-root=$CVC_PATH  --save-path $SAVEPATH
python predict.py --train-dataset=Kvasir --test-dataset=CVC --data-root=$CVC_PATH  --save-path $SAVEPATH
python predict.py --train-dataset=CVC --test-dataset=Kvasir --data-root=$KVASIR_PATH  --save-path $SAVEPATH
python predict.py --train-dataset=Kvasir --test-dataset=Kvasir --data-root=$KVASIR_PATH  --save-path $SAVEPATH

python eval.py --train-dataset=CVC --test-dataset=CVC --data-root=$CVC_PATH  --save-path $SAVEPATH
python eval.py --train-dataset=Kvasir --test-dataset=CVC --data-root=$CVC_PATH  --save-path $SAVEPATH
python eval.py --train-dataset=CVC --test-dataset=Kvasir --data-root=$KVASIR_PATH  --save-path $SAVEPATH
python eval.py --train-dataset=Kvasir --test-dataset=Kvasir --data-root=$KVASIR_PATH  --save-path $SAVEPATH