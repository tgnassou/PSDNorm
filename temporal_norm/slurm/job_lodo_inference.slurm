#!/bin/bash

## JOB INFO
#SBATCH --job-name=lodo_psdnorm
#SBATCH --output=slurm_logs/lodo_psdnorm.out
#SBATCH --error=slurm_logs/lodo_psdnorm.err

## NODE CONFIGURATION
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=02:00:00

cd $WORK/test_time_temporal_normalization/temporal_norm/
## ENV ACTIVATION

module load pytorch-gpu/py3/2.6.0

## CODE EXECUTION
srun python run_LODO_inference.py --dataset MROS --percent 0.15