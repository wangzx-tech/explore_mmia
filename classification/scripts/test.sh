python3 classification/predict.py\
         --data_dir datas/classification/lld-mmri-2023/images \
         --val_anno_file datas/classification/lld-mmri-2023/labels/labels_test.txt \
         --model ple_fusion \
         --batch-size 4 \
         --checkpoint classification/output/ple_fusion/model_best.pth.tar \
         --results-dir classification/output/ple_fusion/ \



