#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --time=00:55:00
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
#conda list tensorflow

export SLURM_MPI_TYPE=pmi2

PRT_DIR="/scratch/bbtb"

imgFolderTXT="${PRT_DIR}/textshape50.txt"
modelFolder="resnet18-layer-blur-category-50-sigma-1.5-2024-12-14-17-30-11"
modelP="${PRT_DIR}/zhenans2/CV_bandpass_adv/training_outputs/${modelFolder}/ckpt_epk40.pth"

srun python ${PRT_DIR}/zhenans2/CV_bandpass_adv/human-vis-freq-align/attack.py ${PRT_DIR}/imagenet \
        --img-folder-txt $imgFolderTXT \
		--save-dir "." \
		--model-pth $modelP \
		--arch "resnet18" \
		--num-category 50 \
		--append-layer "blur" \
		--custom-sigma 1.5 \
		--lp "linf" \
		--attack-alg "pgd" \
		--seed 415 \
		--batch-size 256 \
		--workers 4

# --kernel-size 31 \
# --custom-sigma 2.0 \
exit 
