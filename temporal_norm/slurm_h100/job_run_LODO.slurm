#!/bin/bash

## JOB INFO
#SBATCH --job-name=lodo_lodo_psdnorm
#SBATCH --output=slurm_logs/lodo_all_datasets_%j.out
#SBATCH --error=slurm_logs/lodo_all_datasets_%j.out

## NODE CONFIGURATION
#SBATCH -A wbb@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --time=10:00:00
#SBATCH --array=0-39

## ENV ACTIVATION
module purge
module load arch/h100
module load pytorch-gpu/py3/2.6.0

cd $WORK/PSDNorm/temporal_norm

## PARAMETER LISTS
DATASETS=("ABC" "CCSHS" "CFS" "CHAT" "HOMEPAP" "MASS" "MROS" "PhysioNet" "SHHS" "SOF")
NORMS=("PSDNorm" "BatchNorm")
MODEL_NAMES=("USleep" "DeepSleepNet")

NUM_DATASETS=${#DATASETS[@]}
NUM_NORMS=${#NORMS[@]}
NUM_MODELS=${#MODEL_NAMES[@]}

TOTAL_COMBINATIONS=$((NUM_DATASETS * NUM_NORMS * NUM_MODELS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_COMBINATIONS" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, max allowed is $((TOTAL_COMBINATIONS - 1))"
    exit 1
fi

# Decode task ID into dataset, norm, and model
DATASET_ID=$(( SLURM_ARRAY_TASK_ID % NUM_DATASETS ))
NORM_ID=$(( (SLURM_ARRAY_TASK_ID / NUM_DATASETS) % NUM_NORMS ))
MODEL_ID=$(( SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_NORMS) ))

DATASET="${DATASETS[$DATASET_ID]}"
NORM="${NORMS[$NORM_ID]}"
MODEL_NAME="${MODEL_NAMES[$MODEL_ID]}"

echo "Running: Dataset=$DATASET | Norm=$NORM | Model=$MODEL_NAME"

## CODE EXECUTION
srun python run_LODO.py \
    --dataset "$DATASET" \
    --percent 1 \
    --norm "$NORM" \
    --model_name "$MODEL_NAME"
