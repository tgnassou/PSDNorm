#!/bin/bash

## JOB INFO
#SBATCH --job-name=lodo_psdnorm
#SBATCH --output=slurm_logs/lodo_psdnorm.out
#SBATCH --error=slurm_logs/lodo_psdnorm.err

## NODE CONFIGURATION
#SBATCH -A wbb@h100
#SBATCH --constraint=h100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --time=20:00:00

## ENV ACTIVATION
module purge

module load arch/h100
module load pytorch-gpu/py3/2.6.0

cd $WORK/PSDNorm/temporal_norm

## CODE EXECUTION
srun python run_LODO.py --dataset CHAT --percent 1 --use_amp
