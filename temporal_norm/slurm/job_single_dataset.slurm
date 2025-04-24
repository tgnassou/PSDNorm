#!/bin/bash

## JOB INFO
#SBATCH --job-name=CareSleepNet
#SBATCH --output=slurm_logs/CareSleepNet.out
#SBATCH --error=slurm_logs/CareSleepNet.err

## NODE CONFIGURATION
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=10:00:00

cd $WORK/test_time_temporal_normalization/temporal_norm/
## ENV ACTIVATION

module load pytorch-gpu/py3/2.6.0

## CODE EXECUTION
srun python run_LODO.py --dataset CHAT --percent 0.01 --model_name CareSleepNet --batch_size 32 --norm BatchNorm