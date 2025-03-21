#!/bin/bash

#SBATCH --job-name=all_covariance
#SBATCH --output=parallel-job-output.log
#SBATCH --error=parallel-job-error.log
#SBATCH --mail-user=daniel.johnson@umontpellier.fr
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --time=5-00:00:00
#SBATCH --array=0-255
#SBATCH --mem=5000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --dependency=singleton

source ~/lenstronomyenv/bin/activate

# Record start time
start_time=$(date +%s)

# Read task from task list
TASK_FILE="tasks.txt"
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $TASK_FILE)

# Run the corresponding task
python -u job.py $TASK

# Record end time
end_time=$(date +%s)

# Compute total runtime and log it
runtime=$((end_time - start_time))
echo "Job $SLURM_ARRAY_TASK_ID completed in $runtime seconds" >> runtime.log