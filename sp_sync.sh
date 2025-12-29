#!/bin/bash 
#SBATCH --job-name=sp_sync 
# SBATCH --output=/home/%u/.out/%j.out ## this is where print() etc. go -> $HOME/.out 
# SBATCH --error=/home/%u/.out/%j.err ## this is where errors go -> $HOME/.out 
#SBATCH --time=0-24:00:00 ## max. time in format d-hh:mm:ss 
#SBATCH --nodes=1 ## number of nodes, usually 1 in python 
#SBATCH --mem-per-cpu=500MB ## specify the memory per core 
# #SBATCH --mem=500MB ## alternatively, specify the memory (commented) 
#SBATCH --ntasks=1 ## number of tasks, usually 1 in python 
#SBATCH --cpus-per-task=20 ## number of cores 
#SBATCH --partition=defq ## queue (partition) to run the job in 
# #SBATCH --partition=qjupyter ## alternative queue (commented) 
# #SBATCH --nodelist=srv-lab-t-251 ## run on a specific worker (commented)exit 
# #SBATCH --account=my_special_project ## account to charge the job to (commented)

set -euo pipefail

# create output directory (doesn't do anything if it already exists)
mkdir -p ${HOME}/.out

# set up environment variable with parent directory of this script
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

# -------------------- CONFIG --------------------
# New remote (AWARENew) – reviewed ET flat folder
REMOTE_REVIEWED='zhaw_aware_new'
REMOTE_REVIEWED_PATH='Data Sep 2025'  

# Local destinations
LOCAL_OLD_TREE='/store/kruu/eye_tracking/training_data_before_review'
LOCAL_WORK_TREE='/store/kruu/eye_tracking/training_data'      # raw training data location
LOCAL_REVIEWED_FLAT='/store/kruu/eye_tracking/reviewed_et_flat'       # where flat reviewed TSVs go

mkdir -p "$LOCAL_REVIEWED_FLAT"

# -------------------- SYNC OLD TREE (DEPRECIATED) --------------------
# Old remote (AWARE) – full tree incl ASD/simulator/etc.
# REMOTE_OLD='zhaw'
# REMOTE_OLD_PATH='Shared Documents/Data Sept 2025'   

# echo "[info] Syncing OLD dataset tree (ASD/simulator/etc.)"
# rclone lsd "${REMOTE_OLD}:" || true
# rclone lsd "${REMOTE_OLD}:${REMOTE_OLD_PATH}" || { echo "[error] Old remote path not found"; exit 1; }

# rclone sync "${REMOTE_OLD}:${REMOTE_OLD_PATH}" "${LOCAL_TREE}" \
#   --checkers 8 --transfers 4 --tpslimit 8 --tpslimit-burst 16 \
#   --retries 10 --retries-sleep 10s --low-level-retries 20 \
#   --contimeout 1m --timeout 10m \
#   --log-file "${HOME}/.out/rclone_pull_old_${SLURM_JOB_ID:-local}.log" --log-level INFO

# -------------------- COPY OLD TREE--------------------
# Copy old tree -> working tree (one-time or per-run depending on what you want)
# If LOCAL_WORK_TREE exists and you want a fresh copy each run, wipe it first.
# if [[ -d "$LOCAL_WORK_TREE" ]]; then
#   echo "[info] Removing existing work tree: $LOCAL_WORK_TREE"
#   rm -rf "$LOCAL_WORK_TREE"
# fi

# echo "[info] Copying old tree -> work tree"
# cp -a "$LOCAL_OLD_TREE" "$LOCAL_WORK_TREE"


# -------------------- SYNC REVIEWED ET (FLAT) --------------------
echo "[info] Syncing REVIEWED flat ET"
rclone lsd "${REMOTE_REVIEWED}:" || true
rclone lsd "${REMOTE_REVIEWED}:${REMOTE_REVIEWED_PATH}" || { echo "[error] Reviewed remote path not found"; exit 1; }

rclone sync "${REMOTE_REVIEWED}:${REMOTE_REVIEWED_PATH}" "${LOCAL_REVIEWED_FLAT}" \
  --checkers 8 --transfers 4 --tpslimit 8 --tpslimit-burst 16 \
  --retries 10 --retries-sleep 10s --low-level-retries 20 \
  --contimeout 1m --timeout 10m \
  --log-file "${HOME}/.out/rclone_pull_reviewed_${SLURM_JOB_ID:-local}.log" --log-level INFO

# -------------------- MATERIALIZE REVIEWED ET INTO TREE --------------------
echo "[info] Materializing reviewed ET into local tree"
# writes/symlinks to: LOCAL_WORK_TREE/<pid>/Scenario <sid>/ET/reviewed_gaze_data_fusion.tsv
uv run -m utils.materialize_reviewed_et "${LOCAL_REVIEWED_FLAT}" "${LOCAL_WORK_TREE}" symlink

# -------------------- BUILD RAW INPUTS (PROCESS ET + ASD) --------------------
echo "[info] Running build_raw_inputs on tree"
# Takes new reviewed ET file simlinked before, or original ET file if not exist.
uv run -m utils.build_raw_inputs "${LOCAL_WORK_TREE}"