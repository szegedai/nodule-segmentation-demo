#!/bin/bash
#SBATCH --job-name=nodule-eval
#SBATCH --account=ainoduleseg
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=182G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

# Metrics evaluation launcher.
#
# Usage:
#   sbatch slurm/eval.sh nodule configs/segresnet_small.yaml checkpoints/segresnet_small/best.pth
#   sbatch slurm/eval.sh joint  configs/joint_segresnet_pseudo.yaml checkpoints/joint_segresnet_pseudo/best.pth
# Optional: SPLIT=test (default: val). Site-specific parts: see slurm/train.sh.

TASK="${1:?usage: sbatch slurm/eval.sh <nodule|joint> <config.yaml> <ckpt.pth>}"
CONFIG="${2:?config required}"
CKPT="${3:?checkpoint required}"
SPLIT="${SPLIT:-val}"
DATA_ROOT="${DATA_ROOT:?export DATA_ROOT=/path/to/dataset before submitting}"
SIF="${SIF:-nodule-seg.sif}"

case "$TASK" in
  nodule) SCRIPT=eval_metrics_nodule.py ;;
  joint)  SCRIPT=eval_metrics_joint.py ;;
  *) echo "unknown task: $TASK (expected nodule|joint)"; exit 1 ;;
esac

echo "Started on $(hostname) at $(date)"
echo "task=$TASK config=$CONFIG ckpt=$CKPT split=$SPLIT"
mkdir -p logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DATA_ROOT="$DATA_ROOT" \
apptainer exec --nv --bind "$DATA_ROOT" "$SIF" \
    python3 "$SCRIPT" --config "$CONFIG" --ckpt "$CKPT" --split "$SPLIT"

echo "Finished at $(date)"
