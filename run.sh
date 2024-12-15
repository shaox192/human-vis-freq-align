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

# attack-alg: "natual"
# severity: for natual attack, strength with which to corrupt on image; an integer in [0, 5]
# perturb-type: one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    # 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    # 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    # 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    # the last four are validation functions
python attack.py $toyData\
    --img-folder-txt $toyCats \
    --model-pth $modelP \
    --arch "resnet18" \
    --num-category 50 \
    --append-layer "bandpass" \
    --kernel-size 31 \
    --custom-sigma 2.0 \
    --lp "linf" \
    --attack-alg "natural"\
    --severity 3\
    --perturb-type "gaussian_noise"\
    --seed 415 \
    --batch-size 3 \
    --workers 0\
