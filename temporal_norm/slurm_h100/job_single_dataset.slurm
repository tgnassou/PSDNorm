#!/bin/bash

## JOB INFO
#SBATCH --job-name=lodo_psdnorm_
#SBATCH --output=slurm_logs/lodo_psdnorm_phys.out
#SBATCH --error=slurm_logs/lodo_psdnorm_phys.err

## NODE CONFIGURATION
#SBATCH -A chr@h100
#SBATCH --constraint=h100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=20:00:00

## ENV ACTIVATION
module purge

module load arch/h100
module load pytorch-gpu/py3/2.6.0

cd $WORK/PSDNorm/temporal_norm

## CODE EXECUTION
srun python run_LODO_balanced.py \
    --dataset "PhysioNet" \
    --balanced_acc \
    --model_name "USleep" \
    --norm "LayerNorm" \
    --filter_size "0" \
    --num_workers 24 \
    --batch_size 64 \
    --compile \
    --seed "0" \
    --filter_size_reduce \
    --torchinductor \
    --results_path "results_balanced" \
    --n_epochs 30 \
    --n_subjects "5730"