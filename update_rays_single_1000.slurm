#!/bin/bash
#SBATCH --job-name=update_rays-single-1000
#SBATCH -p gpu-mxian
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=update_rays-single-1000-%j.log
set -euo pipefail

module load py-jupyterlab/4.0.1
cd ~/SURF2025

BASE_OUT=./update_rays-single_results_1000
mkdir -p "$BASE_OUT"
echo "Writing RayS-single outputs (5 samples) to $BASE_OUT"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for ds in CIFAR10 MNIST ImageNet; do
  OUTDIR="$BASE_OUT/$ds"
  mkdir -p "$OUTDIR"
  echo "Starting RayS on $ds (1000 samples)..."
  python main_update.py \
    --attack RayS \
    --dataset "$ds" \
    --num_samples 1000 \
    --output_dir "$OUTDIR" \
    --seed 0
  echo "Finished RayS-single on $ds → results in $OUTDIR"
done

echo "All RayS-single for 1000 samples completed. Check $BASE_OUT for per-dataset results."