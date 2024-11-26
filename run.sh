#!/bin/bash

which python

toyData="/Users/zhenanshao/Documents/git_repos/ML_toydata/ImageNet"
toyCats="/Users/zhenanshao/Documents/git_repos/ML_toydata/ImageNet/toy.txt"
python train.py $toyData\
    --save_dir . \
    --img_folder_txt $toyCats \
    --category-209 \
    --arch resnet18 \
    --pretrained \
    --append-layer "bandpass" \
    --kernel-size 31 \

#   --orig-imagenet-lbs ../annot/imagenet_labels.txt \
#   --orig-manifold-stats ../reprs/manifolds_train_stats/mftma_orig_sub1_tunedNP_train_V1.pkl \
#   --save_dir . \
#   --roi $ROI \
#   --arch resnet18 \
#   --seed 415 \
#   --pretrained \
#   --batch-size 3 \
#   --lr 0.01 \
#   --epochs 2 \
#   --save-interval 5 \
#   --print-freq 1 \
#   --alphas 0.7 0.15 0.15 0 0
