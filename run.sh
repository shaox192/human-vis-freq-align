#!/bin/bash

which python


############# -------------- train ---------------
## Uncomment this section to train. remember to comment out the attack section

# toyData="/home/chengxiao/cs543/final_project/toydata/ImageNet"
# toyCats="/home/chengxiao/cs543/final_project/toydata/ImageNet/toy.txt"
# python train.py $toyData\
#     --save-dir "outputs" \
#     --img-folder-txt $toyCats \
#     --num-category 2 \
#     --arch "resnet18" \
#     --pretrained \
#     --append-layer "bandpass" \
#     --kernel-size 31 \
#     --custom-sigma 2.0 \
#     --seed 415 \
#     --batch-size 3 \
#     --lr 0.01 \
#     --epochs 2 \
#     --save-interval 1 \
#     --print-freq 1 \


############# -------------- attack ---------------
## Uncomment this section to attack. remember to comment out the train section

toyData="/Users/zhenanshao/Documents/git_repos/ML_toydata/ImageNet"
toyCats="/Users/zhenanshao/Documents/git_repos/ML_toydata/ImageNet/toy.txt"
modelP="./outputs/resnet18-layer-bandpass-category-2-sigma-2.0-2024-12-03-22-17-48/ckpt_epk1.pth"

python attack.py $toyData\
    --img-folder-txt $toyCats \
    --model-pth $modelP \
    --arch "resnet18" \
    --num-category 2 \
    --append-layer "bandpass" \
    --kernel-size 31 \
    --custom-sigma 2.0 \
    --lp "linf" \
    --attack-alg "fgsm" \
    --seed 415 \
    --batch-size 3 \
    --workers 0
