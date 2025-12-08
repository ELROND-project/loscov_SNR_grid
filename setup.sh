#!/bin/bash
#SBATCH --job-name=init_job
#SBATCH --output=logs/init_%j.log
#SBATCH --error=logs/init_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=5000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=1-00:00:00

ORIGINAL_DIR=$(pwd)  #the current working directory

echo "Running initialization scripts"
source ~/lenstronomyenv/bin/activate
python -u first_steps.py
python -u ccov.py
echo "Initialization finished"
