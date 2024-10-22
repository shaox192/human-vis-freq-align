#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --time=00:45:00
#SBATCH --account=bbtb-delta-gpu
#SBATCH --job-name=MANIFOLD
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
SUB_ID="sub1"
ROI=V1
origImageNetLbs="${PRT_DIR}/zhenans2/BIAIfollowup/manifolds/BIAImanifolds/annot/imagenet_labels.txt"
manStatsF="${PRT_DIR}/zhenans2/BIAIfollowup/manifolds/manifold_stats/mftma/mftma_${SUB_ID}_tunedNP_train_${ROI}.pkl"
manStatsOrig="${PRT_DIR}/zhenans2/BIAIfollowup/manifolds/manifold_stats/mftma/mftma_orig_${SUB_ID}_tunedNP_train_${ROI}.pkl"

# save_dir="${PRT_DIR}/zhenans2/BIAIfollowup/manifolds/ckpt/manifold_orig_dim_V1_0.1"
save_dir="${PRT_DIR}/zhenans2/BIAIfollowup/manifolds/ckpt/manifold_orig_comb_V1"

srun python ${PRT_DIR}/zhenans2/BIAIfollowup/manifolds/BIAImanifolds/training/cotrain_man.py ${PRT_DIR}/imagenet \
        --sub ${SUB_ID} \
        --img_folder_txt ${PRT_DIR}/coco50.txt \
        --save_dir ${save_dir} \
        --manifold-stats ${manStatsF}  \
        --orig-imagenet-lbs ${origImageNetLbs} \
        --orig-manifold-stats ${manStatsOrig} \
	--decorr-ON \
        --roi $ROI \
        --arch resnet18 \
        --seed 415 \
        --pretrained \
        --batch-size 1024 \
        --lr 0.1 \
	--weight-decay 0.0001 \
        --epochs 56 \
        --save-interval 5 \
        --print-freq 1 \
        --alphas 0.6 0.2 0.2 0.0 0.0\
        --multiprocessing-distributed \
        --dist-url 'tcp://127.0.0.1:2000' --dist-backend 'nccl' --world-size 1 --rank 0
        

exit 

