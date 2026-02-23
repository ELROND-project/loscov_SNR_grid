#!/bin/bash
#SBATCH --job-name=generate_params
#SBATCH --output=logs/generate_params.log
#SBATCH --error=logs/generate_params.err
#SBATCH --ntasks=1
#SBATCH --mem=5000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=1-00:00:00

ORIGINAL_DIR=$(pwd)  #the current working directory

echo "Generating param file"
source ~/lenstronomyenv/bin/activate
python -u generate_params.py
echo "File generated"
