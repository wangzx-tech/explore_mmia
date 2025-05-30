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

Pretrained models can found in [google drive]([https://drive.google.com/drive/folders/1Dq9pjWID-1FKrISXIRfGajtE2WKkgLEu?usp=drive_link](https://drive.google.com/file/d/1oX-gdpvfYN8f1PNS6DA2SS0hs--P4jdO/view?usp=sharing)).

### for training 
```
bash classification/scripts/train.sh
```

### for testing
```
bash classification/scripts/test.sh
```


