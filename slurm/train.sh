#!/bin/bash
#SBATCH --job-name=nodule-train
#SBATCH --account=ainoduleseg
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=182G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00

# Generic training launcher for this repo (ROI / nodule / joint — the task is
# read from the config's `task:` field).
#
# Usage:
#   sbatch --job-name=<tag> slurm/train.sh configs/<config>.yaml
#
# Site-specific parts (adjust for your cluster):
#   --account/--partition/--cpus-per-task/--mem  : SZTE supercomputer values;
#     the per-GPU bundle is NOT applied automatically — request it explicitly.
#   DATA_ROOT : root of the preprocessed dataset (see README "Data layout").
#   SIF       : Apptainer image built from container/nodule-seg.def.

CONFIG="${1:?usage: sbatch slurm/train.sh <config.yaml>}"
DATA_ROOT="${DATA_ROOT:?export DATA_ROOT=/path/to/dataset before submitting}"
SIF="${SIF:-nodule-seg.sif}"

echo "Started on $(hostname) at $(date)"
echo "alloc: nproc=$(nproc)  cpus_on_node=$SLURM_CPUS_ON_NODE  mem_per_node=${SLURM_MEM_PER_NODE:-n/a}MB  gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed "s/^/alloc: gpu=/"
echo "config: $CONFIG"
mkdir -p logs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DATA_ROOT="$DATA_ROOT" \
apptainer exec --nv --bind "$DATA_ROOT" "$SIF" \
    python3 train.py --config "$CONFIG" --resume

echo "Finished at $(date)"
