#!/bin/bash
#SBATCH --job-name=lodo_norm
#SBATCH --output=slurm_logs/lodo_norm_%A_%a.out
#SBATCH --error=slurm_logs/lodo_norm_%A_%a.err

#SBATCH -A wbb@v100
#SBATCH --ntasks=1
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-3

module purge
module load pytorch-gpu/py3/2.6.0

cd "$WORK/PSDNorm/temporal_norm"

NORMS=(PSDNorm BatchNorm PSDNorm BatchNorm)
MODELS=(CNNTransformer CNNTransformer USleep USleep)

NORM=${NORMS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running with normalization: $NORM and model: $MODEL"

# Launch your training
srun python run_LODO.py \
    --norm "$NORM" \
    --model_name "$MODEL" \
    --num_workers 20 \
    $( [ "$MODEL" = "CNNTransformer" ] && echo "--lr 1e-4" ) \
    $( [ "$MODEL" = "USleep" ] && echo "--lr 1e-3" ) \
    --percent 1 \
    --dataset CHAT \
    --batch_size 64 \
    --compile \
    $( [ "$NORM" = "PSDNorm" ] && echo "--filter_size 9" )
