#!/bin/bash
#SBATCH --job-name=instance-metrics
#SBATCH --account=ainoduleseg
#SBATCH --output=logs/instance_%j.out
#SBATCH --error=logs/instance_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=182G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

# Instance-level (per-nodule) recall/precision. Usage:
#   sbatch slurm/instance_metrics.sh <config.yaml> <ckpt.pth> [output.json]
# Optional env: SPLIT (default test), DATA_ROOT, SIF — see slurm/train.sh.

CONFIG="${1:?usage: sbatch slurm/instance_metrics.sh <config.yaml> <ckpt.pth> [out.json]}"
CKPT="${2:?checkpoint required}"
OUT="${3:-}"
SPLIT="${SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:?export DATA_ROOT=/path/to/dataset before submitting}"
SIF="${SIF:-nodule-seg.sif}"

echo "Started on $(hostname) at $(date)"
echo "config=$CONFIG ckpt=$CKPT split=$SPLIT"
mkdir -p logs

CMD="python3 scripts/instance_metrics.py --config $CONFIG --ckpt $CKPT --split $SPLIT"
if [ -n "$OUT" ]; then CMD="$CMD --output $OUT"; fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DATA_ROOT="$DATA_ROOT" \
apptainer exec --nv --bind /project/AINoduleSeg "$SIF" $CMD

echo "Finished at $(date)"
