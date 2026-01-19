#!/bin/bash
#SBATCH --job-name=ncov
#SBATCH --output=logs/ncov.log
#SBATCH --error=logs/ncov.err
#SBATCH --ntasks=1
#SBATCH --mem=500
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=10-00:00:00


ORIGINAL_DIR=$(pwd)

# Number of parameter combinations to run per array task
PARAMS_PER_TASK=20  # Adjust this number based on your needs

# Count total parameter combinations
NPARAMS=$(awk 'NF' "${ORIGINAL_DIR}/params.txt" | wc -l) #count only non-empty lines

# Calculate number of array tasks needed
NTASKS=$(( (NPARAMS + PARAMS_PER_TASK - 1) / PARAMS_PER_TASK ))

# If no array task ID set, resubmit with proper array
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Total parameters: $NPARAMS"
    echo "Parameters per task: $PARAMS_PER_TASK"
    echo "Submitting job array with $NTASKS tasks (max 400 concurrent)..."
    sbatch --array=0-$((NTASKS-1))%400 "$0"
    exit 0
fi

###############################################################
# Process multiple parameter combinations in this task
###############################################################

# Calculate which lines this task should process
START_LINE=$(( SLURM_ARRAY_TASK_ID * PARAMS_PER_TASK + 1 ))
END_LINE=$(( START_LINE + PARAMS_PER_TASK - 1 ))

# Don't exceed total number of parameters
if [ "$END_LINE" -gt "$NPARAMS" ]; then
    END_LINE=$NPARAMS
fi

echo "Task $SLURM_ARRAY_TASK_ID processing lines $START_LINE to $END_LINE"

source ~/lenstronomyenv/bin/activate

# Loop through assigned parameter combinations
for LINE_NUM in $(seq $START_LINE $END_LINE); do
    PARAMS=$(awk 'NF' "${ORIGINAL_DIR}/params.txt" | sed -n "${LINE_NUM}p")
    SIGMA_L=$(echo "$PARAMS" | awk '{print $1}')
    NLENS=$(echo "$PARAMS" | awk '{print $2}')
    
    echo "  Line $LINE_NUM: sigma_L = $SIGMA_L, Nlens = $NLENS"
    python -u ncov.py "$SIGMA_L" "$NLENS"
done

echo "Task $SLURM_ARRAY_TASK_ID completed"