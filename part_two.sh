#!/bin/bash
#SBATCH --job-name=part_two
#SBATCH --output=logs/part_two.log
#SBATCH --error=logs/part_two.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=10-00:00:00
#SBATCH --array=0-5%6
# adjust 0-5 if tasks.txt length changes

set -euo pipefail #stop everything if something fails

ORIGINAL_DIR=$(pwd)
TASKS_FILE="${ORIGINAL_DIR}/tasks.txt"

# Read the task corresponding to this array index (skip empty lines)
TASK=$(awk 'NF {print $1}' "$TASKS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

if [ -z "$TASK" ]; then
    echo "No task found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "SLURM task $SLURM_ARRAY_TASK_ID running task: $TASK"

source ~/lenstronomyenv/bin/activate

python -u part_two_interpolations.py "$TASK"

echo "Task $TASK completed"