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
                        --num-classes 21 \
                        --split-id ./splits/voc/split_0.pkl
``` 
### Validation 
```
python evaluate.py --dataset pascal_voc  \
                   --num-classes 21 \
                   --restore-from ./checkpoints/voc_semi_0_125/VOC_30000.pth 
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

## Instructions for setting-up Multi-Label Mean-Teacher branch
This work is based on the [Mean-Teacher](https://arxiv.org/abs/1703.01780) Semi-supervised Learning work. To use the MLMT branch, follow the instructions below. 
1. Fork the [mean-teacher](https://github.com/CuriousAI/mean-teacher) repo. 
2. Modify the fully connected layer, according to the number of classes and add Sigmoid activation for multi-label classification.
3. Use Binary Cross Entropy loss fucntion instead of multi-class Cross entropy. 
4. Load the pretrained ImageNet weights for ResNet-101 from ```./pretrained_models/```.
5. Use student/teacher predictions for Network output fusion with s4GAN branch. 
6. For lower labeled-ratio, early stopping might be required.  


## Acknowledgement

Parts of the code have been adapted from: 
[DeepLab-Resnet-Pytorch](https://github.com/speedinghzl/Pytorch-Deeplab), [AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg), [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)


## Citation

```
@article{1908.05724,
    Author = {Sudhanshu Mittal and Maxim Tatarchenko and Thomas Brox},
    Title = {Semi-Supervised Semantic Segmentation with High- and Low-level Consistency},
    journal = {arXiv preprint arXiv:1908.05724},
    Year = {2019},
}
```

