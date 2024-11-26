#!/bin/bash

which python

toyData="/Users/zhenanshao/Documents/git_repos/ML_toydata/ImageNet"
toyCats="/Users/zhenanshao/Documents/git_repos/ML_toydata/ImageNet/toy.txt"
python train.py $toyData\
    --save-dir "outputs" \
    --img-folder-txt $toyCats \
    --category-209 \
    --arch "resnet18" \
    --pretrained \
    --append-layer "bandpass" \
    --kernel-size 31 \
    --seed 415 \
    --batch-size 3 \
    --lr 0.01 \
    --epochs 2 \
    --save-interval 1 \
    --print-freq 1 \
