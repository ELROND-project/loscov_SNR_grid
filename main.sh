#!/bin/bash

#SBATCH --job-name='all_covariance'
#SBATCH --output=main-stdout.log
#SBATCH --error=main-stderr.log
#SBATCH --mail-user=daniel.johnson@umontpellier.fr
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=256
#SBATCH --mem=5000
#SBATCH --partition=lupm

source ~/lenstronomyenv/bin/activate
echo "Submitting Slurm job"
python -u main.py