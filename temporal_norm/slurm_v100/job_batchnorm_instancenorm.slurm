#!/bin/bash
#SBATCH --job-name=lodo_norm
#SBATCH --output=slurm_logs/lodo_norm_%A_%a.out
#SBATCH --error=slurm_logs/lodo_norm_%A_%a.err

#SBATCH -A chr@v100
#SBATCH --ntasks=1
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=2:00:00
#SBATCH --array=0-7

module purge
module load pytorch-gpu/py3/2.6.0

cd $WORK/PSDNorm/temporal_norm

## PARAMETER LISTS
DATASETS=("ABC" "CCSHS")
NORMS=("BatchNorm" "InstanceNorm")
MODEL_NAMES=("USleep")
SEEDS=(0 1)

NUM_DATASETS=${#DATASETS[@]}
NUM_NORMS=${#NORMS[@]}
NUM_MODELS=${#MODEL_NAMES[@]}
NUM_SEEDS=${#SEEDS[@]}

TOTAL_COMBINATIONS=$((NUM_DATASETS * NUM_NORMS * NUM_MODELS * NUM_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_COMBINATIONS" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, max allowed is $((TOTAL_COMBINATIONS - 1))"
    exit 1
fi

# Decode task ID into dataset, filter, model and percent
DATASET_ID=$(( SLURM_ARRAY_TASK_ID % NUM_DATASETS ))
NORM_ID=$(( (SLURM_ARRAY_TASK_ID / NUM_DATASETS) % NUM_NORMS ))
MODEL_ID=$(( (SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_NORMS)) % NUM_MODELS ))
SEED_ID=$(( SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_NORMS * NUM_MODELS) ))

DATASET="${DATASETS[$DATASET_ID]}"
NORM="${NORMS[$NORM_ID]}"
MODEL_NAME="${MODEL_NAMES[$MODEL_ID]}"
SEED="${SEEDS[$SEED_ID]}"

echo "Running: Dataset=$DATASET | Norm=$NORM | Model=$MODEL_NAME | Seed=$SEED"

# CODE EXECUTION
srun python run_LODO.py \
    --dataset "$DATASET" \
    --model_name "$MODEL_NAME" \
    --norm "$NORM" \
    --filter_size 0 \
    --percent "0.25" \
    --num_workers 10 \
    $( [ "$MODEL_NAME" = "CNNTransformer" ] && echo "--lr 1e-4" ) \
    $( [ "$MODEL_NAME" = "USleep" ] && echo "--lr 1e-3" ) \
    --batch_size 64 \
    --compile \
    --seed "$SEED" \
    --torchinductor

