#!/bin/bash

which python


############# -------------- train ---------------
## Uncomment this section to train. remember to comment out the attack section
# saveSuffix="sigma-1.5"
# toyData="/Users/zhenanshao/Documents/git_repos/ML_toydata/ImageNet"
# toyCats="/Users/zhenanshao/Documents/git_repos/ML_toydata/ImageNet/toy.txt"
# python train.py $toyData\
#     --save-dir "outputs" \
#     --save-suffix $saveSuffix \
#     --img-folder-txt $toyCats \
#     --num-category 2 \
#     --arch "resnet18" \
#     --pretrained \
#     --append-layer "blur" \
#     --kernel-size 31 \
#     --custom-sigma 1.5 \
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
modelP="./outputs/resnet18-layer-blur-category-2-sigma-1.5-2024-12-14-14-00-42/ckpt_epk1.pth"

python attack.py $toyData\
    --img-folder-txt $toyCats \
    --model-pth $modelP \
    --arch "resnet18" \
    --num-category 2 \
    --append-layer "blur" \
    --kernel-size 31 \
    --custom-sigma 1.5 \
    --lp "linf" \
    --attack-alg "fgsm" \
    --seed 415 \
    --batch-size 3 \
    --workers 0
