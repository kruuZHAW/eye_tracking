#!/bin/bash 
#SBATCH --job-name=data_sync 
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
    SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
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

# TODO: MAKE THIS WORK ONCE ACCESS TO TOULOUSE
# Optional: where original data are located (if need to refresh it)
# "\\toulouse\hf-datastorage\AWARE\Sept 2025"
# If this is a Windows UNC path,  apparently need it mounted on Linux first.
# Example mount point: /mnt/toulouse/AWARE/Sept_2025
BASE_TREE_REMOTE_MOUNT='/mnt/toulouse/AWARE/Sept_2025'            # <-- change to actual mount path
BASE_TREE_MODE='use_local'  # 'use_local' | 'sync_from_remote' (use local copy on the cluster or the one on Toulouse)

mkdir -p "$LOCAL_REVIEWED_FLAT"
mkdir -p "$LOCAL_WORK_TREE"
mkdir -p "$LOCAL_REVIEWED_FLAT"

# -------------------- PREPARE BASE TREE --------------------
# Decide SOURCE for the base tree
case "$BASE_TREE_MODE" in
  use_local)
    BASE_SRC="${LOCAL_OLD_TREE}"
    ;;
  sync_from_remote)
    BASE_SRC="${BASE_TREE_REMOTE_MOUNT}"
    ;;
  *)
    echo "[error] Unknown BASE_TREE_MODE=$BASE_TREE_MODE"
    exit 1
    ;;
esac

# Always recreate LOCAL_WORK_TREE from BASE_SRC
echo "[info] Rebuilding work tree from base source (tsv + db only): ${BASE_SRC}"
rm -rf "${LOCAL_WORK_TREE}"
mkdir -p "${LOCAL_WORK_TREE}"

rsync -a --prune-empty-dirs \
  --include='*/' \
  --include='*/ET/***' \
  --include='*/simulator/***' \
  --include='*.tsv' \
  --include='*.db' \
  --include='*.zip' \
  --exclude='*' \
  --exclude='*/Training/***' \
  "${BASE_SRC}/" "${LOCAL_WORK_TREE}/"

echo "[debug] Copied TSV count: $(find "${LOCAL_WORK_TREE}" -type f -name '*.tsv' | wc -l)"
echo "[debug] Copied DB  count: $(find "${LOCAL_WORK_TREE}" -type f -name '*.db'  | wc -l)"
echo "[debug] Example scenario dirs:"
find "${LOCAL_WORK_TREE}" -maxdepth 2 -type d -name "Scenario *" | head

# Basic sanity check: do we see participant directories?
if ! find "$LOCAL_WORK_TREE" -maxdepth 2 -type d -name "Scenario *" | head -n 1 >/dev/null; then
  echo "[warn] LOCAL_WORK_TREE doesn't look like a scenario tree. Check BASE_TREE_MODE / mount path."
fi

# -------------------- SYNC REVIEWED ET (FLAT) --------------------
echo "[info] Syncing REVIEWED flat ET"
rclone lsd "${REMOTE_REVIEWED}:" || true
rclone lsd "${REMOTE_REVIEWED}:${REMOTE_REVIEWED_PATH}" || { echo "[error] Reviewed remote path not found"; exit 1; }

rclone copy "${REMOTE_REVIEWED}:${REMOTE_REVIEWED_PATH}" "${LOCAL_REVIEWED_FLAT}" \
  --checkers 8 --transfers 4 --tpslimit 8 --tpslimit-burst 16 \
  --retries 10 --retries-sleep 10s --low-level-retries 20 \
  --contimeout 1m --timeout 10m \
  --log-file "${HOME}/.out/rclone_pull_reviewed_${SLURM_JOB_ID:-local}.log" --log-level INFO

# -------------------- MATERIALIZE REVIEWED ET INTO TREE --------------------
echo "[info] Materializing reviewed ET into local tree"
# writes/symlinks to: LOCAL_WORK_TREE/<pid>/Scenario <sid>/ET/reviewed_gaze_data_fusion.tsv
uv run -m utils.materialize_reviewed_et "${LOCAL_REVIEWED_FLAT}" "${LOCAL_WORK_TREE}" copy

# -------------------- BUILD RAW INPUTS (PROCESS ET + ASD) --------------------
echo "[info] Running build_raw_inputs on tree"
# Takes new reviewed ET file simlinked before, or original ET file if not exist.
uv run -m utils.build_raw_inputs "${LOCAL_WORK_TREE}"