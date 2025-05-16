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
#SBATCH --cpus-per-task=24
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
FILTER_SIZE=(0 1 9)
MODEL_NAMES=("USleep")
SEEDS=(0 1 2)

NUM_DATASETS=${#DATASETS[@]}
NUM_FILTERS=${#FILTER_SIZE[@]}
NUM_MODELS=${#MODEL_NAMES[@]}
NUM_SEEDS=${#SEEDS[@]}

TOTAL_COMBINATIONS=$((NUM_DATASETS * NUM_FILTERS * NUM_MODELS * NUM_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_COMBINATIONS" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, max allowed is $((TOTAL_COMBINATIONS - 1))"
    exit 1
fi

# Decode task ID into dataset, filter, model and percent
DATASET_ID=$(( SLURM_ARRAY_TASK_ID % NUM_DATASETS ))
FILTER_ID=$(( (SLURM_ARRAY_TASK_ID / NUM_DATASETS) % NUM_FILTERS ))
MODEL_ID=$(( (SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_FILTERS)) % NUM_MODELS ))
SEED_ID=$(( SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_FILTERS * NUM_MODELS) ))

DATASET="${DATASETS[$DATASET_ID]}"
FILTER="${FILTER_SIZE[$FILTER_ID]}"
MODEL_NAME="${MODEL_NAMES[$MODEL_ID]}"
SEED="${SEEDS[$SEED_ID]}"

echo "Running: Dataset=$DATASET | Filter=$FILTER | Model=$MODEL_NAME | Seed=$SEED"

# CODE EXECUTION
srun python run_LODO.py \
    --dataset "$DATASET" \
    --model_name "$MODEL_NAME" \
    --filter_size "$FILTER" \
    --percent "0.2" \
    --num_workers 10 \
    $( [ "$MODEL_NAME" = "CNNTransformer" ] && echo "--lr 1e-4" ) \
    $( [ "$MODEL_NAME" = "USleep" ] && echo "--lr 1e-3" ) \
    --batch_size 64 \
    --compile \
    --torchinductor \
    --seed "$SEED" \
