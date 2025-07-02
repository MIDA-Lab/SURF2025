#!/bin/bash
#SBATCH --job-name=update_signopt_lf-1000
#SBATCH -p gpu-mxian
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=update_signopt_lf-1000-%j.log
set -euo pipefail

# Load required modules
module load py-jupyterlab/4.0.1
cd ~/SURF2025

# Setup output directory
BASE_OUT=./update_signopt_lf_results_1000
mkdir -p "$BASE_OUT"
echo "Writing Sign-OPT_lf outputs (1000 samples) to $BASE_OUT"

# Set environment variables
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run attacks for each dataset
for ds in CIFAR10 MNIST ImageNet; do
    OUTDIR="$BASE_OUT/$ds"
    mkdir -p "$OUTDIR"
    
    echo "Starting Sign-OPT on $ds (1000 samples)..."
    
    # Add timing and error handling
    start_time=$(date +%s)
    if python main_update.py \
        --attack Sign-OPT \
        --dataset "$ds" \
        --num_samples 1000 \
        --output_dir "$OUTDIR" \
        --seed 0; then
        
        end_time=$(date +%s)
        runtime=$((end_time - start_time))
        echo "Finished Sign-OPT on $ds → results in $OUTDIR (took ${runtime}s)"
    else
        echo "ERROR: Sign-OPT failed on $ds" >&2
        exit 1
    fi
done

echo "All Sign-OPT_lf for 1000 samples completed. Check $BASE_OUT for per-dataset results."