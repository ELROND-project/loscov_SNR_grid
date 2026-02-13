#!/bin/bash
#SBATCH --job-name=part_three
#SBATCH --output=logs/part_three_%A_%a.out
#SBATCH --error=logs/part_three_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --mem=500
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=10-00:00:00

ORIGINAL_DIR=$(pwd)

# Number of parameter combinations per array task
PARAMS_PER_TASK=32

# Count total non-empty parameter lines
NPARAMS=$(awk 'NF' "${ORIGINAL_DIR}/params.txt" | wc -l)

# Calculate number of array tasks needed
NTASKS=$(( (NPARAMS + PARAMS_PER_TASK - 1) / PARAMS_PER_TASK ))

# If not running as an array job, resubmit properly
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Total parameters: $NPARAMS"
    echo "Parameters per task: $PARAMS_PER_TASK"
    echo "Submitting job array with $NTASKS tasks (max 999 concurrent)..."
    sbatch --array=0-$((NTASKS-1))%999 "$0"
    exit 0
fi

###############################################################
# Process multiple parameter combinations in this task
###############################################################

START_LINE=$(( SLURM_ARRAY_TASK_ID * PARAMS_PER_TASK + 1 ))
END_LINE=$(( START_LINE + PARAMS_PER_TASK - 1 ))

if [ "$END_LINE" -gt "$NPARAMS" ]; then
    END_LINE=$NPARAMS
fi

echo "Array job $SLURM_ARRAY_JOB_ID | Task $SLURM_ARRAY_TASK_ID"
echo "Processing lines $START_LINE to $END_LINE"

source ~/lenstronomyenv/bin/activate

START_TIME=$(date +%s)

for LINE_NUM in $(seq $START_LINE $END_LINE); do
    PARAMS=$(awk 'NF' "${ORIGINAL_DIR}/params.txt" | sed -n "${LINE_NUM}p")
    SIGMA_L=$(echo "$PARAMS" | awk '{print $1}')
    NLENS=$(echo "$PARAMS" | awk '{print $2}')

    echo "Line $LINE_NUM: sigma_L=$SIGMA_L, Nlens=$NLENS"
    python -u part_three_optimisation.py "$SIGMA_L" "$NLENS"
done

END_TIME=$(date +%s)
RUNTIME_SEC=$((END_TIME - START_TIME))
RUNTIME_MIN=$((RUNTIME_SEC / 60))

# Safe runtime logging (atomic)
{
  flock -x 201
  echo "Job $SLURM_ARRAY_JOB_ID | Task $SLURM_ARRAY_TASK_ID | Lines $START_LINE-$END_LINE | Runtime ${RUNTIME_MIN} min (${RUNTIME_SEC} sec)" \
    >> logs/part_three_runtime.log
} 201>/tmp/part_three_runtime.lock

echo "Task $SLURM_ARRAY_TASK_ID completed"