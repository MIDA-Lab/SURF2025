#!/bin/bash
#SBATCH --job-name=update_op_lf-1000
#SBATCH -p gpu-mxian
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=update_opt_lf-1000-%j.log
set -euo pipefail

module load py-jupyterlab/4.0.1
cd ~/SURF2025

BASE_OUT=./update_opt_results_lf_1000
mkdir -p "$BASE_OUT"
echo "Writing OPT outputs (1000 samples) to $BASE_OUT"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for ds in CIFAR10 MNIST ImageNet; do
  OUTDIR="$BASE_OUT/$ds"
  mkdir -p "$OUTDIR"
  echo "Starting OPT on $ds (1000 samples)..."
  python main_update.py \
    --attack OPT \
    --dataset "$ds" \
    --num_samples 1000 \
    --output_dir "$OUTDIR" \
    --seed 0
  echo "Finished OPT on $ds → results in $OUTDIR"
done

echo "All OPT_lf for 1000 samples completed. Check $BASE_OUT for per-dataset results."