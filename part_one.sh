#!/bin/bash
#SBATCH --job-name=part_one
#SBATCH --output=logs/part_one.log
#SBATCH --error=logs/part_one.err
#SBATCH --ntasks=1
#SBATCH --mem=5000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=1-00:00:00

ORIGINAL_DIR=$(pwd)  #the current working directory

echo "Running part one"
source ~/lenstronomyenv/bin/activate
python -u part_one_correlations_distributions.py
echo "Part one finished"
