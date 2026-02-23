#!/bin/bash
#SBATCH --job-name=part_four
#SBATCH --output=logs/part_four.log
#SBATCH --error=logs/part_four.err
#SBATCH --ntasks=1
#SBATCH --mem=5000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=1-00:00:00

ORIGINAL_DIR=$(pwd)  #the current working directory

echo "Running part four"
source ~/lenstronomyenv/bin/activate
python -u part_four_simplifying_data.py
echo "Part four finished"
