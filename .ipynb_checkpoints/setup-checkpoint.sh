#!/bin/bash
#SBATCH --job-name=preprocess_array
#SBATCH --output=logs/preprocess_%A_%a.log
#SBATCH --error=logs/preprocess_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem=5000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=1-00:00:00
#SBATCH --array=0-49

ORIGINAL_DIR=$(pwd)  #the current working directory

###############################################################
# 0. Determine number of lines in params.txt and exit if needed
###############################################################
NLINES=$(wc -l < "${ORIGINAL_DIR}/params.txt")

# If fewer than 50 lines, disable tasks above NLINES-1
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NLINES" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no corresponding parameter. Exiting."
    exit 0
fi

###############################################################
# 1. Run "first_steps.py" and "ccov_job.py" EXACTLY ONCE
#    Only array task 0 executes this
###############################################################
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "Running initialization scripts once"
    source ~/lenstronomyenv/bin/activate
    python -u first_steps.py
    python -u ccov.py
    echo "Initialization finished"
fi

# Small barrier to avoid race conditions
sleep 5

###############################################################
# 2. Extract correct parameters for THIS array index
###############################################################
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${ORIGINAL_DIR}/params.txt")
SIGMA_L=$(echo "$PARAMS" | awk '{print $1}')
NLENS=$(echo "$PARAMS" | awk '{print $2}')

echo "Task $SLURM_ARRAY_TASK_ID running cov.py with:"
echo "    sigma_L = $SIGMA_L"
echo "    Nlens   = $NLENS"

source ~/lenstronomyenv/bin/activate

###############################################################
# 3. Run cov.py with this parameter combination
###############################################################
python -u ncov.py "$SIGMA_L" "$NLENS"