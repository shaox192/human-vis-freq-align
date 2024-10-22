#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --time=00:30:00
#SBATCH --account=bbtb-delta-gpu
#SBATCH --job-name=adv
### GPU options ###
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --gpu-bind=verbose,per_task:1
###SBATCH --gpu-bind=none   # <- or closest
module purge # drop modules and explicitly load the ones needed
       # (good job metadata and reproducibility)
module load anaconda3_gpu
module list # job documentation and metadata
echo "job is starting on `hostname`"
which python3

export SLURM_MPI_TYPE=pmi2

PRT_DIR="/scratch/bbtb"
ROI=TO
wd="0.01"
sub=sub7

# modelP=${PRT_DIR}/zhenans2/nsd/cotrain_${sub}_coco50_fc50/resnet18_neural_arch_resnet18_trained_coco50_roi_${ROI}_layer4_epoch_50.pth

# general model: --model_pth  ${PRT_DIR}/zhenans2/nsd/cotrain_outputs_sub1_avg_renormalize/resnet18_neural_arch_resnet18_trained_${DSET}_roi_${ROI}_layer4_epoch_50.pth \

# V1,VO, random, None model with a specific name
# modelP=${PRT_DIR}/zhenans2/nsd/cotrain_outputs_sub1_avg_renormalize/resnet18_neural_arch_resnet18_trained_coco50_resnet_avg_roi_${ROI}_layer4_epoch_50.pth

# models in cotrain folder
# modelP=${PRT_DIR}/zhenans2/nsd/cotrain_outputs_sub1_avg_renormalize/resnet18_neural_arch_resnet18_trained_coco50_roi_${ROI}_layer4_epoch_50.pth

# shuffle models in projects/
# modelP=/projects/bbtb/zhenans2/BIAI_ckpt/cotrain_sub1/resnet18_neural_arch_resnet18_trained_coco50_roi_${ROI}_layer4_epoch_50.pth

# wd models 
# modelP=/projects/bbtb/zhenans2/BIAI_ckpt/cotrain_sub1/resnet18_neural_arch_resnet18_trained_coco50_roi_None_layer4_wd_${wd}_epoch_50.pth

modelP=${PRT_DIR}/zhenans2/BIAIfollowup/manifolds/ckpt/reg_eval_NOscaling/sub1_roi_TO_coco50_fc50_epk_50.pth

srun python ${PRT_DIR}/zhenans2/BIAIfollowup/manifolds/BIAImanifolds/training/attack_adv.py \
	    --roi $ROI \
		--img_folder_txt ${PRT_DIR}/coco50.txt \
		--data ${PRT_DIR}/imagenet \
		--model-pth $modelP \
	    --neural_predictor_pos layer4 \
	    --neural_predictor_arch resnet18 \
	    --arch resnet18 \
	    --workers 8\
	    --batch-size 64 \
	    --attack-type 0
		# --neural-head-scale


exit
