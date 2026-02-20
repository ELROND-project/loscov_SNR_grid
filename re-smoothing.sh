#!/bin/bash
#SBATCH --job-name=re-smoothing
#SBATCH --output=logs/re-smoothing.log
#SBATCH --error=logs/re-smoothing.err
#SBATCH --ntasks=1
#SBATCH --mem=5000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=1-00:00:00

ORIGINAL_DIR=$(pwd)  #the current working directory

echo "Starting smoothing"
source ~/lenstronomyenv/bin/activate
python -u re-smoothing.py
echo "smoothing completed"
