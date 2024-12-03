#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4
#SBATCH --time=04:00:00
#SBATCH --account=bbtb-delta-gpu
#SBATCH --job-name=CVBandpass
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
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

saveD="${PRT_DIR}/zhenans2/CV_bandpass_adv/training_outputs"
imgFolderTXT="${PRT_DIR}/zhenans2/CV_bandpass_adv/human-vis-freq-align/data/human16-209.txt"
# "${PRT_DIR}/coco50.txt"

srun python ${PRT_DIR}/zhenans2/CV_bandpass_adv/human-vis-freq-align/train.py ${PRT_DIR}/imagenet \
        --save-dir $saveD \
        --img-folder-txt $imgFolderTXT \
        --category-209 \
        --arch "resnet18" \
        --pretrained \
        --append-layer "bandpass" \
        --kernel-size 31 \
        --seed 415 \
        --batch-size 1024 \
        --lr 0.01 \
        --weight-decay 0.0001 \
        --epochs 42 \
        --save-interval 5 \
        --print-freq 1 \
        --multiprocessing-distributed \
        --dist-url 'tcp://127.0.0.1:2000' --dist-backend 'nccl' --world-size 1 --rank 0
        
exit 

