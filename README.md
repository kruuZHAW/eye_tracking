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

## Step 1: Data Synchronisation

## Step 2: Data Processing Pipeline

## Output

## Notes and limitations

