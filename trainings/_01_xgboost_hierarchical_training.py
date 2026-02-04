import os
import sys
import json
from pathlib import Path
sys.path.append(str(Path('~/git_folder/eye_tracking/').expanduser()))
# sys.path.append(str(Path('~/git/eye_tracking/').expanduser()))

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from tqdm import tqdm
from collections import Counter

from utils.data_processing_gaze_data import GazeMetricsProcessor
from utils.data_processing_mouse_data import MouseMetricsProcessor
from utils.data_processing_asd_events import ASDEventsMetricsProcessor
from utils.helper import load_and_process_et, load_asd_scenario_data, drop_chunks_with_nan_et

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

import xgboost as xgb
import joblib 
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn import metrics
from sklearn.base import clone

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

IDLE = -1

# ------------------------- HELPER FUNCTION -------------------------

def detect_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if "Recording timestamp" in col]
    if not candidates:
        raise ValueError("No 'Recording timestamp' column found.")
    return candidates[0] 

def extract_tsfresh_features_from_chunks(
    task_chunks: dict[str, pd.DataFrame],
    columns_to_extract: list[str],
    pval_threshold: float = 0.05,
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Extract and select TSFresh features from a dictionary of task chunks.

    Parameters:
    - task_chunks: dict with `id` as key and corresponding df as value
    - columns_to_extract: list of features to use in TSFresh
    - pval_threshold: p-value threshold for relevance filtering
    - n_jobs: number of parallel jobs for TSFresh

    Returns:
    - final_features: DataFrame of filtered, selected features with `id` as index
    """
    
    # Step 0: Build labels per chunk from the ORIGINAL dict
    task_labels = pd.Series(
        {uid: int(df["Task_id"].iloc[0]) for uid, df in task_chunks.items()},
        name="Task_id"
    )

    
    # Step 1: Recombine into one DataFrame
    # Concatenate everything in one single df for TSFresh
    # Rows are dupplicated if we have overlapping tasks
    value_cols = columns_to_extract
    full_df = pd.concat(task_chunks.values(), ignore_index=True)
    timestamp_col = detect_timestamp_column(full_df)
    
    # Keep only what TSFresh needs, ensure numeric dtypes, and sort per chunk
    full_df = full_df[["id", timestamp_col] + value_cols].copy()
    full_df[value_cols] = full_df[value_cols].apply(pd.to_numeric, errors="coerce")
    full_df = full_df.sort_values(["id", timestamp_col])
    
    # If there are duplicate timestamps within an id, collapse them
    full_df = (full_df
            .groupby(["id", timestamp_col], as_index=False)[value_cols]
            .mean())
    
    # Fill remaining NaNs per chunk (id): ffill/bfill then linear interpolate (Chunks with only missing values have been dropped before)
    def _fill_group(g):
        g[value_cols] = (g[value_cols]
                        .ffill()
                        .bfill()
                        .interpolate(method="linear", limit_direction="both"))
        return g

    full_df = full_df.groupby("id", group_keys=False).apply(_fill_group)
    
    # If anything is still NaN, fill with per-id medians, then global fallback
    medians = full_df.groupby("id")[value_cols].transform("median")
    full_df[value_cols] = full_df[value_cols].fillna(medians)
    full_df[value_cols] = full_df[value_cols].fillna(full_df[value_cols].median())
    
    # 4) Assert clean
    leftover = full_df[value_cols].isna().sum().sum()
    assert leftover == 0, f"Still have {leftover} NaNs after cleaning."

    # Step 2: Run TSFresh
    print("Extracting TSFresh features...")
    extracted_features = extract_features(
        full_df,
        column_id="id",
        column_sort=timestamp_col,
        default_fc_parameters=MinimalFCParameters(), #Deactivate if full feature calculation
        n_jobs=n_jobs,
        disable_progressbar=False
    )

    # Step 3: Impute missing features
    impute(extracted_features)

    # Step 4: Relevance filtering
    relevant_features = calculate_relevance_table(extracted_features, task_labels)
    selected_features = relevant_features[relevant_features["p_value"] < pval_threshold]["feature"]
    
    # Step 5: Final filtered feature matrix
    final_features = extracted_features.loc[task_labels.index, selected_features].reset_index(names="id")

    return final_features

def extract_tsfresh_features_from_multiscale_chunks(
    multiscale_chunks: dict[str, dict[str, pd.DataFrame]],
    columns_to_extract: list[str],
    window_keys: tuple[str] = ("short", "mid", "long"),
    pval_threshold: float = 0.05,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    Extract TSFresh features from multi-scale windows.

    Parameters
    ----------
    multiscale_chunks : dict[str, dict[str, pd.DataFrame]]
        { id: { "short": df_short, "mid": df_mid, "long": df_long } }
    columns_to_extract : list[str]
        Columns to use as time series for TSFresh.
    window_keys : tuple[str, ...]
        Names of the scales (keys in the inner dict).
    pval_threshold : float
        p-value threshold for relevance filtering (per scale).
    n_jobs : int
        TSFresh parallel jobs (per scale).

    Returns
    -------
    features_df : DataFrame
        One row per id, with all selected TSFresh features from each scale,
        columns prefixed with e.g. "short_", "mid_", "long_".
    """

    merged_features = None

    for wname in window_keys:
        # Collect all chunks for a given scale
        scale_chunks: dict[str, pd.DataFrame] = {}

        for uid, windows in multiscale_chunks.items():
            if wname not in windows:
                continue
            df = windows[wname]
            if df is None or df.empty:
                continue  

            scale_chunks[uid] = df

        if not scale_chunks:
            continue

        scale_features = extract_tsfresh_features_from_chunks(
            scale_chunks,
            columns_to_extract=columns_to_extract,
            pval_threshold=pval_threshold,
            n_jobs=n_jobs,
        )

        scale_features = scale_features.copy()
        scale_features = scale_features.rename(
            columns={col: f"{wname}_{col}" for col in scale_features.columns if col != "id"}
        )

        if merged_features is None:
            merged_features = scale_features
        else:
            merged_features = merged_features.merge(scale_features, on="id", how="outer")

    if merged_features is None:
        # No scales produced features
        return pd.DataFrame(columns=["id"])

    return merged_features

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[<>\[\]]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

def split_by_participant(df: pd.DataFrame,
                         group_col="participant_id",
                         test_size=0.2,
                         val_size=0.1,
                         random_state=42):
    # Split off test by participant
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    df_trainval = df.iloc[trainval_idx].copy()
    df_test     = df.iloc[test_idx].copy()

    if val_size > 0:
        # Split train vs val from the remaining participants
        val_prop = val_size / (1.0 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=random_state)
        train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval[group_col]))
        df_train = df_trainval.iloc[train_idx].copy()
        df_val   = df_trainval.iloc[val_idx].copy()
        
        parts = {
            "train_parts": sorted(df_train[group_col].unique().tolist()),
            "val_parts": sorted(df_val[group_col].unique().tolist()),
            "test_parts":  sorted(df_test[group_col].unique().tolist()),
        }
        return df_train, df_val, df_test, parts
    
    parts = {
            "train_parts": sorted(df_trainval[group_col].unique().tolist()),
            "test_parts":  sorted(df_test[group_col].unique().tolist()),
        }
    
    return df_trainval, df_test, parts
    
    

def make_xgb_binary(n_jobs=8, random_state=42):
    # First stage: detect idle VS active tasks
    # probabilities for thresholding
    return xgb.XGBClassifier(
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    
def make_xgb_multi(num_class, n_jobs=8, random_state=42):
    # Second stage: if active task -> detect which one
    # predict probabilities, not labels
    return xgb.XGBClassifier(
        tree_method="hist",
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        n_estimators=600,
        min_child_weight=2,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=n_jobs,
        random_state=random_state,
    )

def class_weight_series(y, cap=8.0, gamma=1, min_weight=0.5):
    """
    Compute per-sample weights to handle class imbalance.

    Parameters
    ----------
    y : array-like
        Class labels.
    cap : float
        Maximum weight (upper clip).
    gamma : float
        Exponent for non-linear scaling. 
        - gamma=1.0 -> classic inverse-frequency.
        - gamma<1.0 -> softer weighting.
        - gamma>1.0 -> more aggressive for rare classes but can makes training unstable.
    min_weight : float
        Minimum weight (lower clip), to avoid pushing frequent classes too close to zero.

    Returns
    -------
    weights : np.ndarray
        Per-sample weights.
    """
    y_series = pd.Series(y)
    counts = y_series.value_counts()

    ratio = counts.median() / counts

    w = ratio ** gamma

    w = w.clip(lower=min_weight, upper=cap)

    return y_series.map(w).values

def rolling_mode(series, k=5):
    # odd k recommended
    from scipy.stats import mode
    vals = []
    s = series.to_numpy()
    half = k // 2
    for i in range(len(s)):
        a = max(0, i - half)
        b = min(len(s), i + half + 1)
        vals.append(Counter(s[a:b]).most_common(1)[0][0])
    return pd.Series(vals, index=series.index)

def ema_probs(probs, alpha=0.6):
    """
    probs: np.ndarray shape [T, C]
    EMA smoothing along time axis.
    """
    probs = np.asarray(probs, dtype=float)
    out = np.zeros_like(probs)
    out[0] = probs[0]
    for t in range(1, len(probs)):
        out[t] = alpha * probs[t] + (1 - alpha) * out[t - 1]
    return out

def participant_sequences(df, sort_key="id"):
    """Yield (participant_id, indices_sorted) to apply smoothing per participant."""
    for pid, sub in df.groupby("participant_id"):
        idx = sub.sort_values(sort_key).index
        yield pid, idx

def oof_pred_stageA(estimator, X, y, groups, sample_weight, cv):
    """Out-of-fold predicted probabilities for the positive class."""
    oof = np.zeros(len(y), dtype=float)
    for tr, va in cv.split(X, y, groups):
        m = clone(estimator)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weight[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof

def oof_pred_stageB(estimator, X, y, groups, sample_weight, cv):
    """
    Out-of-fold predicted labels for multiclass (or binary) classification.
    Returns a Series indexed like y.
    """
    oof = pd.Series(index=y.index, dtype=int)
    for tr, va in cv.split(X, y, groups):
        m = clone(estimator)
        m.fit(X.iloc[tr], y.iloc[tr], sample_weight=sample_weight[tr])
        oof.iloc[va] = m.predict(X.iloc[va]).astype(int)
    return oof

def best_threshold_by_f1(y_true, proba, grid=None):
    """Pick threshold maximizing F1 on (y_true, proba)."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    best_idx = int(np.argmax(f1s))
    return float(grid[best_idx]), float(f1s[best_idx])

def metrics_block(
    title,
    y_true,
    y_pred,
    labels=None,
    average=None,
    idle_label=None,
    compute_active_f1=False,
):
    """
    Unified metrics block for:
      - Stage A (binary): default average='binary'
      - Stage B (multiclass): default average='macro'
      - Combined (multiclass with idle): set idle_label=-1 and compute_active_f1=True

    Parameters
    ----------
    title : str
    y_true, y_pred : array-like or Series
    labels : list/array or None
        Class order for report/confusion matrix.
    average : str or None
        F1 averaging. If None, chooses 'binary' when 2 classes else 'macro'.
    idle_label : int/str or None
        Label to exclude for "active-only" metrics (e.g., -1).
    compute_active_f1 : bool
        If True and idle_label is not None, also compute macro-F1 on y_true != idle_label.

    Returns
    -------
    dict with:
      acc, f1, rep, cm, text
      and optionally f1_active when compute_active_f1=True
    """
    # Align as Series to avoid index mismatches
    y_true = pd.Series(y_true)
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.reindex(y_true.index)
    else:
        y_pred = pd.Series(np.asarray(y_pred), index=y_true.index)
    
    if y_pred.isna().any():
        raise ValueError(f"{title}: y_pred contains NaNs after alignment. Index mismatch likely.")

    # Choose default F1 averaging
    n_classes = y_true.nunique(dropna=False)
    if average is None:
        average = "binary" if n_classes == 2 else "macro"

    acc = accuracy_score(y_true, y_pred)
    f1_main = f1_score(y_true, y_pred, average=average)

    rep = classification_report(y_true, y_pred, digits=3, labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out = {"acc": acc, "f1": f1_main, "rep": rep, "cm": cm}

    f1_active = None
    if compute_active_f1:
        if idle_label is None:
            raise ValueError("compute_active_f1=True requires idle_label to be set (e.g., -1).")
        act_mask = (y_true != idle_label)
        if act_mask.any():
            f1_active = f1_score(y_true[act_mask], y_pred[act_mask], average="macro")
        else:
            f1_active = np.nan
        out["f1_active"] = f1_active

    block = []
    block.append(f"=== {title} ===")
    block.append(f"Accuracy: {acc:.3f}")

    if compute_active_f1:
        block.append(f"Macro-F1 (all classes): {f1_main:.3f}")
        block.append(
            f"Macro-F1 (active only): {f1_active:.3f}"
            if f1_active is not None and not np.isnan(f1_active)
            else "Macro-F1 (active only): NaN (no active samples)"
        )
    else:
        # Stage A / Stage B
        block.append(f"F1-score ({average}): {f1_main:.3f}")

    block.append("")
    block.append("Per-class report:")
    block.append(rep)
    block.append("Confusion matrix:")
    block.append(str(cm))
    block.append("")
    out["text"] = "\n".join(block)

    return out

def build_xy(df, leaky_cols, task_col="Task_id", group_col="participant_id"):
    X = df.drop(columns=[c for c in leaky_cols if c in df.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = sanitize_column_names(X).astype("float32")

    y_tasks = df[task_col].astype(int)          # idle == -1
    y_active = (y_tasks != -1).astype(int)      # Stage A label
    groups = df[group_col].astype(str)

    return X, y_tasks, y_active, groups

def combined_probabilities_from_models(X, mA, mB, all_classes, active_classes, idle_label=-1):
    """
    P(idle|x) = 1 - P(active|x)
    P(task=c|x) = P(active|x) * P(task=c|active,x)
    """
    idx = X.index

    pA = pd.Series(mA.predict_proba(X)[:, 1], index=idx)  # P(active|x)

    prob_B = mB.predict_proba(X)     # [N, Cb]
    classes_B = mB.classes_          # task labels per column (your Task_id values)

    pB = pd.DataFrame(0.0, index=idx, columns=active_classes)
    for j, c in enumerate(classes_B):
        if c in pB.columns:
            pB[c] = prob_B[:, j]

    P = pd.DataFrame(0.0, index=idx, columns=all_classes)
    P[idle_label] = 1.0 - pA
    for c in active_classes:
        P[c] = pA * pB[c]

    # numerical safety
    s = P.sum(axis=1).replace(0.0, np.nan)
    P = P.div(s, axis=0).fillna(0.0)

    return P

def smooth_stageB_and_rebuild_P(pA, pB_df, meta_df, all_classes, active_classes, idle_label=-1, alpha_B=0.6, sort_key="id"):
    """
    pA: Series indexed by samples (P(active|x))
    pB_df: DataFrame indexed by samples, columns=active_classes (P(task=c|active,x))
    meta_df: DataFrame indexed by samples with at least ["participant_id", sort_key]
    returns: P_smooth DataFrame indexed by samples, columns=all_classes
    """
    P_smooth = pd.DataFrame(0.0, index=pA.index, columns=all_classes)

    # Apply per participant
    for pid, g in meta_df.loc[pA.index].groupby("participant_id"):
        idx = g.sort_values(sort_key).index

        pA_seq = pA.loc[idx].values
        pB_seq = pB_df.loc[idx, active_classes].values  # [T, C]

        pB_seq_smooth = ema_probs(pB_seq, alpha=alpha_B)

        P_pid = pd.DataFrame(0.0, index=idx, columns=all_classes)
        P_pid[idle_label] = 1.0 - pA_seq
        for j, c in enumerate(active_classes):
            P_pid[c] = pA_seq * pB_seq_smooth[:, j]

        # Renormalize
        P_pid = P_pid.div(P_pid.sum(axis=1), axis=0).fillna(0.0)
        P_smooth.loc[idx] = P_pid

    return P_smooth

if __name__ == "__main__":
    # ------------------------- 0. PARAMETERS -------------------------
    
    # Directories
    # store_dir = str(Path('~/store/aware').expanduser())
    store_dir = "/store/kruu/eye_tracking"
    data_dir = os.path.join(store_dir, "training_data")
    
    save_model_path = "logs/xgboost_hierarchical_v5"
    os.makedirs(save_model_path, exist_ok=True)
    
    split_names = ["train", "val", "test"]
    save_splits_dir = os.path.join(store_dir, "splits")
    save_split_dirs = [os.path.join(save_splits_dir, split_name) for split_name in split_names]
    
    # Temporary storage for better I/O performance
    # Use that to load data if on GPU cluster
    # temp_data_dir = "/scratch/aware"
    # temp_raw_inputs_dir = os.path.join(temp_data_dir, "training_data_raw_inputs")
    # temp_splits_dir = os.path.join(temp_data_dir, "splits")
    # temp_split_dirs = [os.path.join(temp_splits_dir, split_name) for split_name in split_names]
    
    #Features names
    features = ['Recording timestamp [ms]', 'epoch_ms', 'Gaze point X [DACS px]', 'Gaze point Y [DACS px]', 'Event']
    interpolate_cols = ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]']
    fill_columns = ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]']
    columns_to_extract = ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]'] # Columns for TSFresh
    
    # ------------------------- 1. LOADING DATASET -------------------------
    
    # Loading data from scratch
    chunks_et, blinks, atco_task_map = load_and_process_et(root_dir = data_dir,
                                                            columns = features,
                                                            interpolate_cols = interpolate_cols,
                                                            fill_cols = fill_columns,
                                                            window_short_ms = 5000,
                                                            window_mid_ms = 10000,
                                                            window_long_ms = 25000,
                                                            task_margin_ms = 2000,
                                                            step_ms = 3000,
                                                            filter_outliers = True,
                                                            # participants=["001", "002", "003"],
                                                            time_resampling=False)
    
    print("Number of total occurences per task for all data: ")
    task_window_counts = Counter(int(key.split('_')[2]) for key in chunks_et.keys())
    for task_id in sorted(task_window_counts):
        print(f"Task {task_id}: {task_window_counts[task_id]} windows")
    
    # Dropping id that have too many nans for the gaze
    cleaned_chunks = drop_chunks_with_nan_et(chunks_et, threshold=0.8)
    
    print("Number of total occurences per task after drop: ")
    task_window_counts = Counter(int(key.split('_')[2]) for key in cleaned_chunks.keys())
    for task_id in sorted(task_window_counts):
        print(f"Task {task_id}: {task_window_counts[task_id]} windows")

    # ------------------------- 2. MANUAL FEATURE EXTRACTION -------------------------
    
    asd_scenarios = load_asd_scenario_data(root_dir=data_dir)
    prepared_asd = {}
    for key, df_sc in asd_scenarios.items():
        asd = df_sc.sort_values("epoch_ms").set_index("epoch_ms")
        prepared_asd[key] = asd
    
    window_keys = ("short", "mid", "long")
    rows = []
    
    print("Extracting handmade features...")
    for uid, windows in tqdm(cleaned_chunks.items()):

        row = {"id": uid}
        row["participant_id"] = windows["short"]["Participant name"].iloc[0]
        row["Task_id"] = windows["short"]["Task_id"].iloc[0]
        
        p_s_id = "_".join(uid.split("_")[:2])
        scenario_asd = prepared_asd.get(p_s_id)
        if scenario_asd is None:
            continue

        for wname in window_keys:
            chunk = windows[wname]
            min_epoch = chunk["epoch_ms"].min()
            max_epoch = chunk["epoch_ms"].max()
            # fast time slice thanks to index on epoch_ms
            window_asd = scenario_asd.loc[min_epoch:max_epoch].reset_index()
            
            # Gaze metrics
            gaze_processor = GazeMetricsProcessor(chunk, timestamp_unit="ms")
            gaze_metrics = gaze_processor.compute_all_metrics()   
            prefixed_gaze = {f"{wname}_{k}": v for k, v in gaze_metrics.items()}
            row.update(prefixed_gaze)
            
            # Mouse metrics
            mouse_processor = MouseMetricsProcessor(window_asd, resample=False)
            mouse_metrics = mouse_processor.compute_all_metrics()   
            prefixed_mouse = {f"{wname}_{k}": v for k, v in mouse_metrics.items()}
            row.update(prefixed_mouse)
            
            # ASD events metrics
            asd_processor = ASDEventsMetricsProcessor(window_asd)
            asd_metrics = asd_processor.compute_all_metrics()   
            prefixed_asd = {f"{wname}_{k}": v.iat[0] for k, v in asd_metrics.items()}
            row.update(prefixed_asd)

        rows.append(row)
    
    metrics_df = pd.DataFrame(rows)

    # ------------------------- 3. AUTOMATIC FEATURE EXTRACTION -------------------------
    
    # Run tsfresh feature extraction
    print("Extracting TSFresh...")
    tsfresh_data = extract_tsfresh_features_from_multiscale_chunks(
        cleaned_chunks, 
        ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]'], 
        pval_threshold=0.05, 
        n_jobs=50)
    # ------------------------- 4. BUILD DATASET -------------------------
    
    xgboost_data = metrics_df.merge(tsfresh_data, on="id", how="inner")
    
    # Train/test split (no val needed) 
    train_df, test_df, parts = split_by_participant(xgboost_data, group_col="participant_id",
                                                        test_size=0.2, val_size=0.0, random_state=42)
    
    for d in save_split_dirs:
        os.makedirs(d, exist_ok=True)
    
    print("Saving XGboost train/test datasets")
    train_df.to_parquet(os.path.join(save_split_dirs[0], "train_xgboost.parquet"))
    test_df.to_parquet(os.path.join(save_split_dirs[2], "test_xgboost.parquet"))
    
    # print("Load XGboost train/test datasets for debug")
    # train_df = pd.read_parquet(os.path.join(save_split_dirs[0], "train_xgboost.parquet"))
    # test_df = pd.read_parquet(os.path.join(save_split_dirs[2], "test_xgboost.parquet"))
    
    # CROSS VALIDATION SPLIT BASED ON PARTICIPANT ID
    # columns that must not go into X
    leaky = [
        "id",               # chunk identifier
        "participant_id",   # used as groups
        "Task_id",          # the label
        "Scenario_id",      # metadata
        "Task_execution",   # metadata
    ]
    X_train, y_train_tasks, y_train_active, groups_train = build_xy(train_df, leaky)
    X_test,  y_test_tasks,  y_test_active,  groups_test  = build_xy(test_df,  leaky)

    # Define Group K-Fold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ------------------------- 5. MODEL A: IDLE VS ACTIVE TASK -------------------------
    clf_A = make_xgb_binary(n_jobs=16)
    w_A = class_weight_series(y_train_active, cap=6.0, gamma = 1)
    
    param_dist_A = {
        "n_estimators": randint(300, 900),      
        "max_depth": randint(3, 9),             
        "learning_rate": uniform(0.03, 0.2),  # WARNING: Uniform(a,b) samples in [a, a+b)
        "subsample": uniform(0.5, 0.3),         
        "colsample_bytree": uniform(0.5, 0.3), 
    }

    gs_A = RandomizedSearchCV(
        estimator=clf_A,
        param_distributions=param_dist_A,
        n_iter=100,                   
        scoring="average_precision", # or "roc_auc"                 
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=False,
    )
    
    gs_A.fit(X_train, y_train_active, groups=groups_train, sample_weight=w_A)
    best_A = gs_A.best_estimator_
    pd.DataFrame(gs_A.cv_results_).to_csv(os.path.join(save_model_path, "cv_results_stageA.csv"), index=False)
    
    #### MODEL A REPORT ####
    
    # Compute probabilities for best model on out-of-fold points (For each CV set, we evaluate on the fold that was not used for training)
    oof_proba_A = oof_pred_stageA(
        estimator=best_A,
        X=X_train,
        y=y_train_active,
        groups=groups_train,
        sample_weight=w_A,
        cv=cv
    )
    
    # Select best decision threshold best on F1-score
    thr_A, oof_best_f1 = best_threshold_by_f1(y_train_active.values, oof_proba_A)
    print(f"Chosen threshold (F1-opt): {thr_A:.3f}  | F1: {oof_best_f1:.3f}")
    with open(os.path.join(save_model_path, "stageA_threshold.txt"), "w") as f:
        f.write(f"{thr_A:.6f}\n")
    
    # Out-of-fold evaluation for the train set
    oof_pred_A = (oof_proba_A >= thr_A).astype(int)
    oof_metrics_A = metrics_block("Stage A OOF", y_train_active.values, oof_pred_A)
    pd.DataFrame(oof_metrics_A["cm"]).to_csv(os.path.join(save_model_path, "cm_stageA_oof_thresholded.csv"), index=False)

    # Fit FINAL model on full training data
    final_A = clone(best_A)
    final_A.fit(X_train, y_train_active, sample_weight=w_A)
    bundle = {"model": final_A, "threshold": thr_A}
    joblib.dump(bundle, os.path.join(save_model_path, "best_model_stageA.pkl"))
    
    # TEST metrics using same threshold
    test_proba_A = final_A.predict_proba(X_test)[:, 1]
    test_pred_A  = (test_proba_A >= thr_A).astype(int)
    test_metrics_A = metrics_block("Stage A TEST (using OOF threshold)", y_test_active.values, test_pred_A)
    
    # Save TEST confusion matrix + predictions
    pd.DataFrame(test_metrics_A["cm"]).to_csv(os.path.join(save_model_path, "cm_stageA_test.csv"), index=False)
    pd.DataFrame({
        "proba": test_proba_A,
        "pred":  test_pred_A,
        "true":  y_test_active.values
    }).to_csv(os.path.join(save_model_path, "stageA_test_predictions.csv"), index=False)

    # 6) Write a single report_stageA.txt containing threshold + both evals
    report_lines = []
    report_lines.append("MODEL: XGBoost Stage A (active detection)")
    report_lines.append("")
    report_lines.append(f"Selected threshold (max F1 on OOF): {thr_A:.6f}")
    report_lines.append(f"OOF F1 at selected threshold:       {oof_best_f1:.3f}")
    report_lines.append("")
    report_lines.append(oof_metrics_A["text"])
    report_lines.append(test_metrics_A["text"])

    report_path = os.path.join(save_model_path, "report_stageA.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Saved: {report_path}")
    print(f"Saved: {os.path.join(save_model_path, 'best_model_stageA.pkl')}")
    print(f"Saved: {os.path.join(save_model_path, 'stageA_threshold.txt')}")
    
    print("\n=== Stage A (Active detection) ===")
    print(
        f"OOF  Acc: {oof_metrics_A['acc']:.3f} | "
        f"F1: {oof_metrics_A['f1']:.3f} | "
        f"Threshold: {thr_A:.3f}"
    )
    print(
        f"TEST Acc: {test_metrics_A['acc']:.3f} | "
        f"F1: {test_metrics_A['f1']:.3f}"
    )
    print(f"Saved: {report_path}")
    
    # ------------------------- 6. MODEL B: IF ACTIVE -> DETECT TASK -------------------------
    
    # Training data for Stage B
    mask_active_tr = (y_train_active == 1) 
    X_train_B = X_train.loc[mask_active_tr]
    y_train_B = y_train_tasks.loc[mask_active_tr]
    groups_train_B = groups_train.loc[mask_active_tr]
    
    # Test data for Stage B
    mask_active_te = (y_test_active == 1) 
    X_test_B = X_test.loc[mask_active_te]
    y_test_B = y_test_tasks.loc[mask_active_te]
    assert X_test_B.index.equals(y_test_B.index), "X_test_B and y_test_B indices do not match!"

    # Model
    num_classes_B = y_train_B.nunique()
    clf_B = make_xgb_multi(num_class=num_classes_B, n_jobs=8)
    w_B = class_weight_series(y_train_B, cap=8.0, gamma= 1.2)
    
    # We need a CV that respects same groups but only for active rows:
    cv_B = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    param_dist_B = {
        "n_estimators": randint(500, 1200),
        "max_depth": randint(3, 9),
        "learning_rate": uniform(0.03, 0.2),
        "subsample": uniform(0.5, 0.3),
        "colsample_bytree": uniform(0.5, 0.3),
        "gamma": uniform(0.0, 0.4),  
    }

    gs_B = RandomizedSearchCV(
        estimator=clf_B,
        param_distributions=param_dist_B,
        n_iter=100,              
        scoring="f1_macro",
        cv=cv_B,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=False,
    )

    gs_B.fit(X_train_B, y_train_B, groups=groups_train_B, sample_weight=w_B)

    best_B = gs_B.best_estimator_
    pd.DataFrame(gs_B.cv_results_).to_csv(os.path.join(save_model_path, "cv_results_stageB.csv"), index=False)
    
    ### Reports Model B ###
    
    # CV evaluation
    oof_pred_B = oof_pred_stageB(
        estimator=best_B,
        X=X_train_B,
        y=y_train_B,
        groups=groups_train_B,
        sample_weight=w_B,
        cv=cv_B
    )
    
    oof_metrics_B = metrics_block("Stage B OOF (Active task recognition)", y_train_B, oof_pred_B)
    pd.DataFrame(oof_metrics_B["cm"]).to_csv(os.path.join(save_model_path, "cm_stageB_oof.csv"), index=False)
    
    # Fit FINAL on full Stage B train
    final_B = clone(best_B)
    final_B.fit(X_train_B, y_train_B, sample_weight=w_B)
    joblib.dump({"model": final_B},
            os.path.join(save_model_path, "best_model_stageB.pkl"))
    
    # Evaluation on TEST
    test_pred_B = pd.Series(final_B.predict(X_test_B).astype(int), index=y_test_B.index)
    # print("NaNs in test_pred_B:", test_pred_B.isna().sum())
    test_metrics_B = metrics_block("Stage B TEST (Active task recognition)", y_test_B, test_pred_B)
    pd.DataFrame(test_metrics_B["cm"]).to_csv(os.path.join(save_model_path, "cm_stageB_test.csv"), index=False)

    report_lines = []
    report_lines.append("MODEL: XGBoost Stage B (task recognition | active only)")
    report_lines.append("")
    report_lines.append(oof_metrics_B["text"])
    report_lines.append(test_metrics_B["text"])

    with open(os.path.join(save_model_path, "report_stageB.txt"), "w") as f:
        f.write("\n".join(report_lines)) 
    
    print("\n=== Stage B (Active task recognition) ===")
    print(
        f"OOF  Acc: {oof_metrics_B['acc']:.3f} | "
        f"Macro-F1: {oof_metrics_B['f1']:.3f}"
    )
    print(
        f"TEST Acc: {test_metrics_B['acc']:.3f} | "
        f"Macro-F1: {test_metrics_B['f1']:.3f}"
    )
    print(f"Saved: {os.path.join(save_model_path, 'report_stageB.txt')}")
    # ------------------------- 7. COMBINED REPORTS -------------------------
    
    # Combined hierarchical CV
    all_classes = np.sort(pd.unique(y_train_tasks))
    active_classes = all_classes[all_classes != IDLE]

    P_oof_list, y_oof_list = [], []

    for tr, va in cv.split(X_train, y_train_active, groups_train):
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        yA_tr = y_train_active.iloc[tr]
        yT_tr, yT_va = y_train_tasks.iloc[tr], y_train_tasks.iloc[va]

        # --- Stage A fit on fold-train ---
        mA = clone(best_A)
        mA.fit(X_tr, yA_tr, sample_weight=w_A[tr])

        # --- Stage B fit on fold-train ACTIVE rows only ---
        tr_active_mask = (yA_tr == 1)
        X_tr_B = X_tr.loc[tr_active_mask]
        y_tr_B = yT_tr.loc[tr_active_mask]

        # compute weights for stage B on this fold's active subset
        wB_tr = class_weight_series(y_tr_B, cap=8.0, gamma=1.2)

        mB = clone(best_B)
        mB.fit(X_tr_B, y_tr_B, sample_weight=wB_tr)

        # --- Combine probabilities on fold-val ---
        P_va = combined_probabilities_from_models(
            X=X_va,
            mA=mA,
            mB=mB,
            all_classes=all_classes,
            active_classes=active_classes,
            idle_label=IDLE
        )

        P_oof_list.append(P_va)
        y_oof_list.append(yT_va)

    # OOF combined probabilities + predictions
    P_oof = pd.concat(P_oof_list).sort_index()
    y_oof = pd.concat(y_oof_list).sort_index()
    yhat_oof = P_oof.idxmax(axis=1).loc[y_oof.index]

    oof_metrics_comb = metrics_block(
        "Combined OOF (unsmoothed)",
        y_oof,
        yhat_oof,
        labels=all_classes,
        idle_label=IDLE,
        compute_active_f1=True, 
    )

    pd.DataFrame(oof_metrics_comb["cm"]).to_csv(
        os.path.join(save_model_path, "cm_combined_oof_unsmoothed.csv"),
        index=False
    )

    # ---- Final models trained on full train (for TEST evaluation) ----
    # Stage A final
    final_A = clone(best_A)
    final_A.fit(X_train, y_train_active, sample_weight=w_A)

    # Stage B final (train on all active rows)
    mask_active_tr = (y_train_active == 1)
    X_train_B = X_train.loc[mask_active_tr]
    y_train_B = y_train_tasks.loc[mask_active_tr]
    w_B = class_weight_series(y_train_B, cap=8.0, gamma=1.2)

    final_B = clone(best_B)
    final_B.fit(X_train_B, y_train_B, sample_weight=w_B)

    # Combined TEST
    P_test = combined_probabilities_from_models(
        X=X_test,
        mA=final_A,
        mB=final_B,
        all_classes=all_classes,
        active_classes=active_classes,
        idle_label=IDLE
    )
    yhat_test = P_test.idxmax(axis=1)
    test_metrics_comb = metrics_block(
        "Combined TEST (unsmoothed)",
        y_test_tasks,
        yhat_test,
        labels=all_classes,
        idle_label=IDLE,
        compute_active_f1=True,
    )

    pd.DataFrame(test_metrics_comb["cm"]).to_csv(
        os.path.join(save_model_path, "cm_combined_test_unsmoothed.csv"),
        index=False
    )

    # ---- Write combined report file (same spirit as stage A/B) ----
    report_lines = []
    report_lines.append("MODEL: Hierarchical Combined (Stage A -> Stage B), unsmoothed")
    report_lines.append("")
    report_lines.append(f"IDLE label: {IDLE}")
    report_lines.append(f"Classes (from TRAIN): {list(all_classes)}")
    report_lines.append("")
    report_lines.append(oof_metrics_comb["text"])
    report_lines.append(test_metrics_comb["text"])

    report_path = os.path.join(save_model_path, "report_combined_unsmoothed.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print("\n=== Combined (unsmoothed) ===")
    print(
        f"OOF  Accuracy: {oof_metrics_comb['acc']:.3f} | "
        f"Macro-F1(all): {oof_metrics_comb['f1']:.3f} | "
        f"Macro-F1(active): {oof_metrics_comb['f1_active']:.3f}"
    )
    print(
        f"TEST Accuracy: {test_metrics_comb['acc']:.3f} | "
        f"Macro-F1(all): {test_metrics_comb['f1']:.3f} | "
        f"Macro-F1(active): {test_metrics_comb['f1_active']:.3f}"
    )
    print(f"Saved: {report_path}")
    
    # ------------------------- 8. TEMPORAL SMOOTHING APPLIED TO COMBINED PREDICTIONS ------------------------- 
    
    # Apply per participant:
    # - Don't smooth stage A as Idle is the majority class and smoothing might discard some active tasks
    # - Smooth Stage-B probabilities with EMA, then argmax.
    # - Rebuild probablilities:
    #   - p(idle | x_t) = 1 - pA
    #   - p(task + c | x_t) = pA*pB_smooth  
    
    alpha_B = 0.6  # smoothing factor for Stage B only

    all_classes = np.sort(pd.unique(y_train_tasks))
    active_classes = all_classes[all_classes != IDLE]

    # Meta needed for smoothing: participant + ordering key (id)
    meta_train = train_df.loc[X_train.index, ["participant_id", "id"]].copy()
    meta_train["participant_id"] = meta_train["participant_id"].astype(str)

    meta_test = test_df.loc[X_test.index, ["participant_id", "id"]].copy()
    meta_test["participant_id"] = meta_test["participant_id"].astype(str)

    # ---- OOF collection: store pA and pB for each fold ----
    pA_oof_list = []
    pB_oof_list = []
    y_oof_list  = []

    for tr, va in cv.split(X_train, y_train_active, groups_train):
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        yA_tr = y_train_active.iloc[tr]
        yT_tr, yT_va = y_train_tasks.iloc[tr], y_train_tasks.iloc[va]

        # Stage A fold model
        mA = clone(best_A).fit(X_tr, yA_tr, sample_weight=w_A[tr])
        pA_va = pd.Series(mA.predict_proba(X_va)[:, 1], index=X_va.index)

        # Stage B fold model trained on ACTIVE rows only
        tr_active_mask = (yA_tr == 1)
        X_tr_B = X_tr.loc[tr_active_mask]
        y_tr_B = yT_tr.loc[tr_active_mask]
        wB_tr  = class_weight_series(y_tr_B, cap=8.0, gamma=1.2)

        mB = clone(best_B).fit(X_tr_B, y_tr_B, sample_weight=wB_tr)

        # Predict conditional probs for ALL val rows
        prob_B_va = mB.predict_proba(X_va)
        classes_B = mB.classes_

        pB_va = pd.DataFrame(0.0, index=X_va.index, columns=active_classes)
        for j, c in enumerate(classes_B):
            if c in pB_va.columns:
                pB_va[c] = prob_B_va[:, j]

        pA_oof_list.append(pA_va)
        pB_oof_list.append(pB_va)
        y_oof_list.append(yT_va)

    # Concatenate OOF
    pA_oof = pd.concat(pA_oof_list).sort_index()
    pB_oof = pd.concat(pB_oof_list).sort_index()
    y_oof  = pd.concat(y_oof_list).sort_index()

    # ---- Smooth OOF: Stage B only, per participant ----
    P_oof_smooth = smooth_stageB_and_rebuild_P(
        pA=pA_oof,
        pB_df=pB_oof,
        meta_df=meta_train,
        all_classes=all_classes,
        active_classes=active_classes,
        idle_label=IDLE,
        alpha_B=alpha_B,
        sort_key="id"
    )

    yhat_oof_smooth = P_oof_smooth.idxmax(axis=1).loc[y_oof.index]

    oof_metrics_smooth = metrics_block(
        f"Combined OOF (smoothed, alpha_B={alpha_B})",
        y_oof,
        yhat_oof_smooth,
        labels=all_classes,
        idle_label=IDLE,
        compute_active_f1=True,
    )

    pd.DataFrame(oof_metrics_smooth["cm"]).to_csv(
        os.path.join(save_model_path, "cm_combined_oof_smoothed.csv"),
        index=False
    )

    # ---- Final models (train full) for TEST ----
    final_A = clone(best_A).fit(X_train, y_train_active, sample_weight=w_A)

    mask_active_tr = (y_train_active == 1)
    X_train_B = X_train.loc[mask_active_tr]
    y_train_B = y_train_tasks.loc[mask_active_tr]
    w_B = class_weight_series(y_train_B, cap=8.0, gamma=1.2)
    final_B = clone(best_B).fit(X_train_B, y_train_B, sample_weight=w_B)

    # ---- TEST pA + pB ----
    pA_test = pd.Series(final_A.predict_proba(X_test)[:, 1], index=X_test.index)

    prob_B_test = final_B.predict_proba(X_test)
    classes_B   = final_B.classes_

    pB_test = pd.DataFrame(0.0, index=X_test.index, columns=active_classes)
    for j, c in enumerate(classes_B):
        if c in pB_test.columns:
            pB_test[c] = prob_B_test[:, j]

    # ---- Smooth TEST Stage-B and rebuild combined probabilities ----
    P_test_smooth = smooth_stageB_and_rebuild_P(
        pA=pA_test,
        pB_df=pB_test,
        meta_df=meta_test,
        all_classes=all_classes,
        active_classes=active_classes,
        idle_label=IDLE,
        alpha_B=alpha_B,
        sort_key="id"
    )

    yhat_test_smooth = P_test_smooth.idxmax(axis=1)

    test_metrics_smooth = metrics_block(
        f"Combined TEST (smoothed, alpha_B={alpha_B})",
        y_test_tasks,
        yhat_test_smooth,
        labels=all_classes,
        idle_label=IDLE,
        compute_active_f1=True,
    )

    pd.DataFrame(test_metrics_smooth["cm"]).to_csv(
        os.path.join(save_model_path, "cm_combined_test_smoothed.csv"),
        index=False
    )

    # ---- Write report (same layout style) ----
    report_lines = []
    report_lines.append("MODEL: Hierarchical Combined (Stage A -> Stage B)")
    report_lines.append("Variant: Stage B EMA smoothing only (Stage A not smoothed)")
    report_lines.append("")
    report_lines.append(f"alpha_B: {alpha_B}")
    report_lines.append(f"IDLE label: {IDLE}")
    report_lines.append(f"Classes (from TRAIN): {list(all_classes)}")
    report_lines.append("")
    report_lines.append(oof_metrics_smooth["text"])
    report_lines.append(test_metrics_smooth["text"])

    report_path = os.path.join(save_model_path, "report_combined_smoothed.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))  
    
    print("\n=== Combined (smoothed) ===")
    print(
        f"OOF  Accuracy: {oof_metrics_smooth['acc']:.3f} | "
        f"Macro-F1(all): {oof_metrics_smooth['f1']:.3f} | "
        f"Macro-F1(active): {oof_metrics_smooth['f1_active']:.3f}"
    )
    print(
        f"TEST Accuracy: {test_metrics_smooth['acc']:.3f} | "
        f"Macro-F1(all): {test_metrics_smooth['f1']:.3f} | "
        f"Macro-F1(active): {test_metrics_smooth['f1_active']:.3f}"
    )
    print(f"Saved: {report_path}")                     
                                                       