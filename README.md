# explore_mmia
Official code for paper "Exploiting Shared and Distinctive Representations for Enhanced Multi-Modality Medical Image Analysis".

## Datasets
LLD-MMRI( [here](https://github.com/LMMMEng/LLD-MMRI-Dataset?tab=readme-ov-file) )

## Data architecture
```
-datas
  -classification
    -lld-mmri-2023
      -images
        -MR0000
          -C+A.nii.gz
          -C+Delay.nii.gz
          -C-pre.nii.gz
          -C+V.nii.gz
          -DWI.nii.gz
          -In Phase.nii.gz
          -Out Phase.nii.gz
          -T2WI.nii.gz
        -......

      -labels
        -labels_train.text
        -labels_val.text
        -labels_test.text
```

## Experiments

### for training 
```
bash classification/scripts/train.sh
```

### for testing
```
bash classification/scripts/test.sh
```


