# bin/bash!
python3 classification/train.py \
       --data_dir datas/classification/lld-mmri-2023/images \
       --train_anno_file datas/classification/lld-mmri-2023/labels/labels_train.txt \
       --val_anno_file datas/classification/lld-mmri-2023/labels/labels_val.txt \
       --batch-size  4 \
       --model ple_fusion \
       --drop 0.5 \
       --lr 1e-4 \
       --warmup-epochs 5 \
       --epochs 6 \
       --gpu 1 \
       --output classification/output
