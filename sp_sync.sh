#!/bin/bash
#SBATCH --job-name=sp_sync
# SBATCH --output=/home/%u/.out/%j.out       ## this is where print() etc. go -> $HOME/.out
# SBATCH --error=/home/%u/.out/%j.err        ## this is where errors go       -> $HOME/.out
#SBATCH --time=0-24:00:00                   ## max. time in format d-hh:mm:ss
#SBATCH --nodes=1                           ## number of nodes, usually 1 in python
#SBATCH --mem-per-cpu=500MB                 ## specify the memory per core
# #SBATCH --mem=500MB                       ## alternatively, specify the memory (commented)
#SBATCH --ntasks=1                          ## number of tasks, usually 1 in python
#SBATCH --cpus-per-task=20                  ## number of cores
#SBATCH --partition=defq                    ## queue (partition) to run the job in
# #SBATCH --partition=qjupyter              ## alternative queue (commented)
# #SBATCH --nodelist=srv-lab-t-251          ## run on a specific worker (commented)exit
# #SBATCH --account=my_special_project      ## account to charge the job to (commented)

set -euo pipefail

# create output directory (doesn't do anything if it already exists)
mkdir -p ${HOME}/.out

# set up environment variable with parent directory of this script
#
# Source: https://stackoverflow.com/questions/56962129/how-to-get-original-location-of-script-used-for-slurm-job
#
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
# `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi
echo "SCRIPT_PATH: $SCRIPT_PATH"

SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_DIR="$SCRIPT_DIR"
echo "PROJECT_DIR: $PROJECT_DIR"

APP_ROOT="$(dirname "$(dirname "$SCRIPT_PATH")")"
echo "APP_ROOT: $APP_ROOT"

# load relevant module
module load mamba

echo "I am running on $SLURM_JOB_NODELIST"
echo "I am running with job id $SLURM_JOB_ID"

# SharePoint sync to fetch datasets

REMOTE='zhaw'
REMOTE_PATH='Data Sept 2025'
LOCAL_IN='/store/kruu/eye_tracking/training_data'
# LOCAL_OUT='/cluster/data/results'

rclone sync "${REMOTE}:${REMOTE_PATH}" "${LOCAL_IN}" \
  --checkers 8 --transfers 4 --tpslimit 8 --tpslimit-burst 16 \
  --retries 10 --retries-sleep 10s --low-level-retries 20 \
  --contimeout 1m --timeout 10m --log-file "${PWD}/rclone_pull.log" --log-level INFO

# # Sharepoint pushback (Maybe pushback the training datasets + the results after the pipeline has been run)
# rclone copy "${LOCAL_OUT}" "${REMOTE}:${REMOTE_PATH}/results" \
#   --checkers 8 --transfers 4 --tpslimit 8 --tpslimit-burst 16 \
#   --retries 10 --retries-sleep 10s --low-level-retries 20 \
#   --contimeout 1m --timeout 10m --log-file "${PWD}/rclone_push.log" --log-level INFO

# uv run "${PROJECT_DIR}/utils/build_raw_inputs.py" "${LOCAL_IN}"
uv run -m utils.build_raw_inputs "${LOCAL_IN}"

