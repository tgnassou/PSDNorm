#!/bin/bash

## JOB INFO
#SBATCH --job-name=lodo_lodo_psdnorm
#SBATCH --output=slurm_logs/lodo_all_datasets_%j.out
#SBATCH --error=slurm_logs/lodo_all_datasets_%j.out

## NODE CONFIGURATION
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=100:00:00
#SBATCH --array=0-9


## ENV ACTIVATION
module purge

module load pytorch-gpu/py3/2.6.0

cd $WORK/experiments/ram/ram_project/
## ENV ACTIVATION

## CODE EXECUTION
DATASETS=("ABC" "CHAT" "CFS" "SHHS" "HOMEPAP" "CCSHS" "MASS" "PhysioNet" "SOF" "MROS")

NUM_DATASETS=${#DATASETS[@]}
DATASET_ID=$(( SLURM_ARRAY_TASK_ID % NUM_DATASETS ))

DATASET="${DATASETS[$DATASET_ID]}"

## MINIMAL TEST
srun python run_LODO.py --dataset $DATASET --percent 0.15