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
The script uses fixed paths (cluster/storage-specific):
- LOCAL_WORK_TREE: the reconstructed training data tree (scenario structure)
- LOCAL_REVIEWED_FLAT: local folder holding the flat reviewed TSVs (for September acquisition)
- REMOTE_REVIEWED + REMOTE_REVIEWED_PATH: location of reviewed eye-tracking files (for September acquisition)
- BASE_TREE_MODE: whether the base scenario tree is pulled from
  - use_local (default): uses the existing local source tree
  - sync_from_remote: uses a mounted remote share (e.g., Toulouse) as the base source (doesn't work directly on Toulouse)

### 1) Materialized reviewed eye-tracking into tree
Module: `utils.materialize_reviewed_et`

Called by:
```
uv run -m utils.materialize_reviewed_et "${LOCAL_REVIEWED_FLAT}" "${LOCAL_WORK_TREE}" copy
```
Reviewed TSVs are provided in a flat structure with filenames like: `<pid>_scenario_<sid>_gaze_data_fusion.tsv`.
The script parses participant (pid) and scenarios (sid) IDs from the filename and writes the file into the scenario tree:
`<LOCAL_WORK_TREE>/<pid>/Scenario <sid>/ET/`.
It outputs files named: `*_reviewed.tsv`. 
Downstream processing automatically prefers `*_reviewed.tsv` over the original eye-tracking files.

### 2) Build raw input parquet files for ET and ASD
Module: `utils.build_raw_inputs`

Called by:
```
uv run -m utils.build_raw_inputs "${LOCAL_WORK_TREE}"
```
This step generates two files per scenario inside `<pid>/Scenario <sid>/taskRecognition/`:
- `raw_et.parquet`
- `raw_asd.parquet`

#### ET processing (`raw_et.parquet`)
- Loads the selected eye-tracking TSV
  - Prefers `*_reviewed.tsv` if present
  - Otherwise falls back to a gaze fusion TSV or any TSV available
- Builds absolute EPOCH timestamps:
  - Uses `Recording date` + `Recording start time` + `Recording timestamp [ms]`
  - Produces a UTC-based millisecond timestamp (`epoch_ms_raw`)
- Applies optional timestamp synchronization:
  - Looks for sync points at: `/store/regd/sync_points/<pid>_scenario_<sid>_sync_points.json`
  - If available, `epoch_ms_synced` is computed and used as the main timeline (`epoch_ms`)
- Removes calibration/out-of-window rows:
  - If present, data is sliced between `ScreenRecordingStart` and `ScreenRecordingEnd`
- Writes the result to `raw_et.parquet`

#### ASD processing (`raw_asd.parquet`)
- Locates the simulator DB (`polaris-events-*.db`) under `simulator/`
  - If only a ZIP is present, it extracts the newest `.db` from the ZIP
- Builds an ASD event dataframe from the DB
- Slices ASD events to the time range covered by the ET recording (from min to max ET `epoch_ms`)
- Writes the result to `raw_asd.parquet` (warns if empty)

### Notes
- The bash script includes `#SBATCH` headers and is run as a Slurm job on the cluster.
- Internally it uses `uv run -m ...` so the Python environment is taken from `pyproject.toml`.

## Step 2: Data Processing Pipeline

## Output

## Notes and limitations

