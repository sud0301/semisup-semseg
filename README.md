# Semi-supevised Semantic Segmentation with High- and Low-level Consistency

This Pytorch repository contains the code for our work [Semi-supervised Semantic Segmentation with High- and Low-level Consistency](https://arxiv.org/pdf/1908.05724.pdf). The approach uses two network branches that link semi-supervised classification with semi-supervised segmentation including self-training. The approach attains significant improvement over existing methods, especially when trained with very few labeled samples. On several standard benchmarks - PASCAL VOC 2012,PASCAL-Context, and Cityscapes - the approach achieves new state-of-the-art in semi-supervised learning.

We propose a two-branch approach to the task of semi-supervised semantic segmentation. The lower branch predicts pixel-wise class labels and is referred to as the Semi-Supervised Semantic Segmentation GAN(s4GAN). The upper branch performs image-level classification and is denoted as the Multi-Label Mean Teacher(MLMT).

Here, this repository contains the source code for the s4GAN branch. MLMT branch is adapted from Mean-Teacher work for semi-supervised classification. Instructions for setting up the MLMT branch are given below. 


## Package pre-requisites
The code runs on Python 3 and Pytorch 0.4 The following packages are required. 

```
pip install scipy tqdm matplotlib numpy opencv-python
```

## Dataset preparation

Download ImageNet pretrained Resnet-101([Link](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)) and place it ```./pretrained_models/```

### PASCAL VOC
Download the dataset([Link](https://lmb.informatik.uni-freiburg.de/resources/datasets/voc_dataset.tar.gz)) and extract in ```./data/voc_dataset/```

### PASCAL Context
Download the annotations([Link](https://lmb.informatik.uni-freiburg.de/resources/datasets/pascal_context_labels.tar.gz)) and extract in ```./data/pcontext_dataset/```

### Cityscapes
Download the dataset from the Cityscapes dataset server([Link](https://www.cityscapes-dataset.com/)). Download the files named 'gtFine_trainvaltest.zip', 'leftImg8bit_trainvaltest.zip' and extract in ```./data/city_dataset/```  

## Training and Validation on PASCAL-VOC Dataset

Results in the paper are averaged over 3 random splits. Same splits are used for reporting baseline performance for fair comparison.

### Training fully-supervised Baseline (FSL)
```
python train_full.py    --dataset pascal_voc  \
                        --checkpoint-dir ./checkpoints/voc_full \
                        --ignore-label 255 \
                        --num-classes 21 
```
### Training semi-supervised s4GAN (SSL)
```
python train_s4GAN.py   --dataset pascal_voc  \
                        --checkpoint-dir ./checkpoints/voc_semi_0_125 \
                        --labeled-ratio 0.125 \
                        --ignore-label 255 \ 
                        --num-classes 21
``` 
### Validation 
```
python evaluate.py --dataset pascal_voc  \
                   --num-classes 21 \
                   --restore-from ./checkpoints/voc_semi_0_125/VOC_30000.pth 
```

### Training MLMT Branch

```
python train_mlmt.py \
        --batch-size-lab 16 \
        --batch-size-unlab 80 \
        --labeled-ratio 0.125 \
        --exp-name voc_semi_0_125_MLMT \
        --pkl-file ./checkpoints/voc_semi_0_125/train_voc_split.pkl
```

### Final Evaluation S4GAN + MLMT
```
python evaluate.py --dataset pascal_voc  \
                   --num-classes 21 \
                   --restore-from ./checkpoints/voc_semi_0_125/VOC_30000.pth \
                   --with-mlmt \
                   --mlmt-file ./mlmt_output/voc_semi_0_125_MLMT/output_ema_raw_100.txt
    
```

## Training and Validation on PASCAL-Context Dataset
```
python train_full.py    --dataset pascal_context  \
                        --checkpoint-dir ./checkpoints/pc_full \
                        --ignore-label -1 \
                        --num-classes 60

python train_s4GAN.py  --dataset pascal_context  \
                       --checkpoint-dir ./checkpoints/pc_semi_0_125 \
                       --labeled-ratio 0.125 \
                       --ignore-label -1 \
                       --num-classes 60 \
                       --split-id ./splits/pc/split_0.pkl
                       --num-steps 60000

python evaluate.py     --dataset pascal_context  \
                       --num-classes 60 \
                       --restore-from ./checkpoints/pc_semi_0_125/VOC_40000.pth
```

## Training and Validation on Cityscapes Dataset
```
python train_full.py    --dataset cityscapes \
                        --checkpoint-dir ./checkpoints/city_full_0_125 \
                        --ignore-label 250 \
                        --num-classes 19 \
                        --input-size '256,512'  

python train_s4GAN.py   --dataset cityscapes \
                        --checkpoint-dir ./checkpoints/city_semi_0_125 \
                        --labeled-ratio 0.125 \
                        --ignore-label 250 \
                        --num-classes 19 \
                        --split-id ./splits/city/split_0.pkl \
                        --input-size '256,512' \
                        --threshold-st 0.7 \
                        --learning-rate-D 1e-5 

python evaluate.py      --dataset cityscapes \
                        --num-classes 19 \
                        --restore-from ./checkpoints/city_semi_0_125/VOC_30000.pth 
```
##Training and validation on Cityscapes with DeepLabv3+ (WRN-38 backbone)
```
python train_s4GAN_wrn38.py   --dataset cityscapes \
                        --checkpoint-dir ./checkpoints/city_semi_v3_wrn38_0_10 \
                        --labeled-ratio 0.10 \
                        --ignore-label 250 \
                        --num-classes 19 \
                        --split-id ./splits/city/split_0.pkl \
                        --input-size '256,512' \
                        --threshold-st 0.55 \
                        --learning-rate-D 1e-5 \
                        --learning-rate 1e-3 \
                        --out results/city_semi_v3_wrn38_0_10
```

## Acknowledgement

Parts of the code have been adapted from: 
[DeepLab-Resnet-Pytorch](https://github.com/speedinghzl/Pytorch-Deeplab), [AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg), [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)


## Citation



```
@ARTICLE{8935407,
  author={S. {Mittal} and M. {Tatarchenko} and T. {Brox}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Semi-Supervised Semantic Segmentation With High- and Low-Level Consistency}, 
  year={2021},
  volume={43},
  number={4},
  pages={1369-1379},
  doi={10.1109/TPAMI.2019.2960224}}
```

