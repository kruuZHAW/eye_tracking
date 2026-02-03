# ATCO Task type prediction pipeline

## Overview
This repository contains a data processing pipeline for task detection using eye-tracking data.
The workflow consists of two main steps:
  1. Synchronizing and preparing the raw data locally
  2. Running a machine-learning pipeline based on XGBoost for feature processing and real-time ATCO task type prediction

The repository can be cloned with:
```
git clone https://github.com/kruuZHAW/eye_tracking.git
cd eye_tracking
```

## Requirements
The project uses `uv` as the Python package manager. All dependencies are defined in the `pyproject.toml` file.
### System requirements 
  - Bash-compatible shell (Linux, macOS, WSL)
  - Python 3.12 (specified in `pyproject.toml`)
  - `uv` package manager
### Installing dependencies
First, install `uv` if it is not already available:
```
pip install uv
```
Then, from the repository root directory, install all required dependencies:
```
uv sync
```
This command will:
  - Create or update the virtual environment
  - Install all dependencies specified in `pyproject.toml`

## Step 1: Data Synchronisation and raw input generation (`data_sync.sh`)

### Run
NB: This script needs an update to deal with the new data loction on `Toulouse`. The current `BASE_TREE_MODE = sync_from_remote` doesn't work. The easiest way is to copy the data onto the cluster, activating `BASE_TREE_MODE = use_local`, and use the copied local directory as `LOCAL_OLD_TREE`.
Run:
```
sbatch data_sync.sh
```
This script prepares the local raw dataset in two phases:
1. Rebuilds a local "work tree" of raw scenarios:
It recreates `LOCAL_WORK_TREE` from a base source (`BASE_SRC`) by copying only the files needed by the pipeline (eye-tracking `.tsv`, polaris simulator `.db`/`.zip`, and required directory structure). This avoids syncing unnecessary subfolders (e.g. video recordings) and guqrantees a clean, reproductible directory tree.
2. Syncs reviewed eye-tracking files (from September acquisition) and injects them imto the scenario tree.
Reviewed eye-tracking TSV files are pulled via `rclone` from a remote location (`REMOTE_REVIEWED_PATH`) and stored locally in a flat folder (`LOCAL_REVIEWED_FLAT`). They are then materialized into the scenario tree so that downstream processing automatically prefers reviewed data when available.

After syncing, the script triggers raw input generation (ET + ASD) for each scenario.

### Configuration paths

## Step 2: Data Processing Pipeline

## Output

## Notes and limitations

