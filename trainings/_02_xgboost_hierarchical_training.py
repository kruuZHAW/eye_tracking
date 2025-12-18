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

    # Split train vs val from the remaining participants
    val_prop = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=random_state)
    train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval[group_col]))
    df_train = df_trainval.iloc[train_idx].copy()
    df_val   = df_trainval.iloc[val_idx].copy()

    # Return the participant IDs per split
    parts = {
        "train_parts": sorted(df_train[group_col].unique().tolist()),
        "val_parts":   sorted(df_val[group_col].unique().tolist()),
        "test_parts":  sorted(df_test[group_col].unique().tolist()),
    }
    return df_train, df_val, df_test, parts

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

def ema_probs(P, alpha=0.6):
    """Exponential smoothing on class-probabilities row-wise per sequence.
       P: np.array shape [T, C] -> smoothed same shape."""
    out = np.empty_like(P)
    out[0] = P[0]
    for t in range(1, len(P)):
        out[t] = alpha * P[t] + (1 - alpha) * out[t-1]
    return out

def participant_sequences(df, sort_key="id"):
    """Yield (participant_id, indices_sorted) to apply smoothing per participant."""
    for pid, sub in df.groupby("participant_id"):
        idx = sub.sort_values(sort_key).index
        yield pid, idx

if __name__ == "__main__":
    # ------------------------- 0. PARAMETERS -------------------------
    
    # Directories
    # store_dir = str(Path('~/store/aware').expanduser())
    store_dir = "/store/kruu/eye_tracking"
    data_dir = os.path.join(store_dir, "training_data")
    
    save_model_path = "/home/kruu/git_folder/eye_tracking/trainings/logs/xgboost_hierarchical_v3"
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
                                                            # participants=["001", "002", "003", "004", "005"],
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
    
    # Redo Train/test/val split as JCAFNET not implemented yet:
    train_df, val_df, test_df, parts = split_by_participant(xgboost_data, group_col="participant_id",
                                                        test_size=0.2, val_size=0.1, random_state=42)
    
    for d in save_split_dirs:
        os.makedirs(d, exist_ok=True)
    
    print("Saving XGboost train/test datasets")
    train_df.to_parquet(os.path.join(save_split_dirs[0], "train_xgboost.parquet"))
    test_df.to_parquet(os.path.join(save_split_dirs[2], "test_xgboost.parquet"))
    val_df.to_parquet(os.path.join(save_split_dirs[1], "val_xgboost.parquet"))
    
    # CROSS VALIDATION SPLIT BASED ON PARTICIPANT ID
    # columns that must not go into X
    leaky = [
        "id",               # chunk identifier
        "participant_id",   # used as groups
        "Task_id",          # the label
        "Scenario_id",      # metadata
        "Task_execution",   # metadata
    ]
    X_train = train_df.drop(columns=[c for c in leaky if c in train_df.columns], errors="ignore")
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_train = sanitize_column_names(X_train).astype("float32")
    
    y_train_tasks = train_df["Task_id"].astype(int) # idle == -1
    is_active = (y_train_tasks != -1).astype(int) 
    groups_train = train_df["participant_id"].astype(str)

    # Define Group K-Fold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ------------------------- 5. MODEL A: IDLE VS ACTIVE TASK -------------------------
    clf_A = make_xgb_binary(n_jobs=16)
    w_A = class_weight_series(is_active, cap=6.0, gamma = 1)
    
    param_dist_A = {
        "n_estimators": randint(300, 900),      
        "max_depth": randint(3, 9),             
        "learning_rate": uniform(0.03, 0.2),   
        "subsample": uniform(0.5, 0.1),         
        "colsample_bytree": uniform(0.5, 0.1), 
    }

    gs_A = RandomizedSearchCV(
        estimator=clf_A,
        param_distributions=param_dist_A,
        n_iter=100,                   
        scoring="f1",                 
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=False,
    )
    
    gs_A.fit(X_train, is_active, groups=groups_train, sample_weight=w_A)
    best_A = gs_A.best_estimator_
    pd.DataFrame(gs_A.cv_results_).to_csv(os.path.join(save_model_path, "cv_results_stageA.csv"), index=False)
    joblib.dump(best_A, os.path.join(save_model_path, "best_model_stageA.pkl"))
    
    #### MODEL A REPORT ####
    accs_A, f1s_A = [], []
    yA_true_all, yA_pred_all = [], []
    for tr, va in cv.split(X_train, is_active, groups_train):
        m = clone(best_A)
        m.fit(X_train.iloc[tr], is_active.iloc[tr], sample_weight=w_A[tr])
        yhat = (m.predict_proba(X_train.iloc[va])[:,1] > 0.5).astype(int)
        accs_A.append(accuracy_score(is_active.iloc[va], yhat))
        f1s_A.append(f1_score(is_active.iloc[va], yhat))
        yA_true_all.append(is_active.iloc[va])
        yA_pred_all.append(pd.Series(yhat, index=is_active.iloc[va].index))
    yA_true_all = pd.concat(yA_true_all).sort_index()
    yA_pred_all = pd.concat(yA_pred_all).loc[yA_true_all.index]
    rep_A = classification_report(yA_true_all, yA_pred_all, digits=3)
    cm_A  = confusion_matrix(yA_true_all, yA_pred_all)
    with open(os.path.join(save_model_path, "report_stageA.txt"), "w") as f:
        f.write(rep_A)
    pd.DataFrame(cm_A).to_csv(os.path.join(save_model_path, "cm_stageA.csv"))
    
    print("\n=== Stage A CV (active detection) ===")
    print(f"Accuracy:                 {np.mean(accs_A):.3f} ± {np.std(accs_A):.3f}")
    print(f"F1-score :   {np.mean(f1s_A):.3f} ± {np.std(f1s_A):.3f}")
    print("\nPer-class report:\n", rep_A)
    
    
    # ------------------------- 6. MODEL B: IF ACTIVE -> DETECT TASK -------------------------
    
    mask_active_tr = (is_active == 1)
    X_train_B = X_train.loc[mask_active_tr]
    y_train_B = y_train_tasks.loc[mask_active_tr]

    num_classes_B = y_train_B.nunique()
    clf_B = make_xgb_multi(num_class=num_classes_B, n_jobs=8)
    w_B = class_weight_series(y_train_B, cap=8.0, gamma= 1.2)
    
    # We need a CV that respects same groups but only for active rows:
    groups_train_B = groups_train.loc[mask_active_tr]
    cv_B = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    param_dist_B = {
        "n_estimators": randint(500, 1200),
        "max_depth": randint(3, 9),
        "learning_rate": uniform(0.03, 0.2),
        "subsample": uniform(0.5, 1),
        "colsample_bytree": uniform(0.5, 1),
        "gamma": uniform(0.0, 0.4),  
    }

    gs_B = RandomizedSearchCV(
        estimator=clf_B,
        param_distributions=param_dist_B,
        n_iter=100,              
        scoring="f1_macro",
        cv=cv_B,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=False,
    )

    gs_B.fit(X_train_B, y_train_B, groups=groups_train_B, sample_weight=w_B)

    best_B = gs_B.best_estimator_
    pd.DataFrame(gs_B.cv_results_).to_csv(os.path.join(save_model_path, "cv_results_stageB.csv"), index=False)
    joblib.dump(best_B, os.path.join(save_model_path, "best_model_stageB.pkl"))
    
    ### Reports Model B ###
    accs_B, f1s_B = [], []
    yB_true_all, yB_pred_all = [], []
    for tr, va in cv_B.split(X_train_B, y_train_B, groups_train_B):
        m = clone(best_B)
        m.fit(X_train_B.iloc[tr], y_train_B.iloc[tr], sample_weight=w_B[tr])
        yhat = m.predict(X_train_B.iloc[va])
        accs_B.append(accuracy_score(y_train_B.iloc[va], yhat))
        f1s_B.append(f1_score(y_train_B.iloc[va], yhat, average="macro"))
        yB_true_all.append(y_train_B.iloc[va])
        yB_pred_all.append(pd.Series(yhat, index=y_train_B.iloc[va].index))
    yB_true_all = pd.concat(yB_true_all).sort_index()
    yB_pred_all = pd.concat(yB_pred_all).loc[yB_true_all.index]
    rep_B = classification_report(yB_true_all, yB_pred_all, digits=3)
    cm_B  = confusion_matrix(yB_true_all, yB_pred_all)
    with open(os.path.join(save_model_path, "report_stageB.txt"), "w") as f:
        f.write(rep_B)
    pd.DataFrame(cm_B).to_csv(os.path.join(save_model_path, "cm_stageB.csv"))
    
    print("\n=== Stage B CV (Active Task recognition) ===")
    print(f"Accuracy:                 {np.mean(accs_B):.3f} ± {np.std(accs_B):.3f}")
    print(f"Macro-F1 :   {np.mean(f1s_B):.3f} ± {np.std(f1s_B):.3f}")
    print("\nPer-class report:\n", rep_B)
    
    # ------------------------- 7. COMBINED REPORTS -------------------------
    
    # Combined hierarchical CV
    # Computes the true probability of each observation of being eahc task.
    accs_comb, f1_macro_comb, f1_macro_active_comb = [], [], []
    
    all_classes = np.sort(y_train_tasks.unique())   
    active_classes = all_classes[all_classes != IDLE]
    
    P_all_unsmoothed = []  # list of DataFrames of combined probabilities for each fold
    Y_all = []  # true labels for each fold
    
    # Store stage A and stage B probabilities for each fold
    pA_list = [] # P(active | x)
    pB_list = [] # P(task = c | active, x) for active + 1 only

    for tr, va in cv.split(X_train, is_active, groups_train):
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        y_tasks_tr, y_tasks_va = y_train_tasks.iloc[tr], y_train_tasks.iloc[va]
        wA_tr = w_A[tr]

        # --- Stage A ---
        mA = clone(best_A).fit(X_tr, is_active.iloc[tr], sample_weight=wA_tr)
        pA = pd.Series(mA.predict_proba(X_va)[:, 1], index=X_va.index)                                        
        
        # --- Stage B (train only on active rows in this fold) ---
        tr_active_mask = (is_active.iloc[tr] == 1)
        mB = clone(best_B).fit(
            X_tr.loc[tr_active_mask],
            y_tasks_tr.loc[tr_active_mask],
            sample_weight=class_weight_series(
                y_tasks_tr.loc[tr_active_mask], 
                cap=8.0, 
                gamma=1.2
            )
        )
        
        # Predict probabilities for all val rows for active classes (not only active)
        prob_B_all = mB.predict_proba(X_va)      # shape [N_va, C_B]
        classes_B = mB.classes_                 # labels corresponding to columns in prob_B_all
        
        # Build full conditional P(task = c | active, x) over all active classes
        pB_fold = pd.DataFrame(
            0.0,
            index=X_va.index,
            columns=active_classes
        )
        for j, c in enumerate(classes_B):
            pB_fold[c] = prob_B_all[:, j]

        # --- Combine ---
        P_fold = pd.DataFrame(
            0.0,
            index=X_va.index,
            columns=all_classes
        )
        
        # Idle probability for each row
        P_fold[IDLE] = 1.0 - pA

        # P(task=c | x) = P(active | x) * P(task=c | active, x) (stage B = marginalization over active r.v.)
        for c in active_classes:
            P_fold[c] = pA * pB_fold[c]
            
        # Numerical safety: renormalize so rows sum to 1
        P_fold = P_fold.div(P_fold.sum(axis=1), axis=0)
        
        # Computing metrics per fold
        yp_fold = P_fold.idxmax(axis=1)
        yt_fold = y_tasks_va
        accs_comb.append(accuracy_score(yt_fold, yp_fold))
        f1_macro_comb.append(f1_score(yt_fold, yp_fold, average="macro"))
        act_mask = (yt_fold != IDLE)
        if act_mask.any():
            f1_macro_active_comb.append(
                f1_score(yt_fold[act_mask], yp_fold[act_mask], average="macro")
            )
            

        P_all_unsmoothed.append(P_fold)
        Y_all.append(yt_fold)
        pA_list.append(pA)
        pB_list.append(pB_fold)
        
    P_all_unsmoothed = pd.concat(P_all_unsmoothed).sort_index()
    yt_all = pd.concat(Y_all).sort_index()
    pA_all = pd.concat(pA_list).sort_index()
    pB_all = pd.concat(pB_list).sort_index()
    
    # Global unsmoothed predictions
    yp_all_unsmoothed = P_all_unsmoothed.idxmax(axis=1)
    
    rep_comb = classification_report(yt_all, yp_all_unsmoothed, digits=3)
    cm_comb  = confusion_matrix(yt_all, yp_all_unsmoothed, labels=all_classes)
    
    print("\n=== Hierarchical CV (Combined unsmoothed) ===")
    print("Final pred counts:", yp_all_unsmoothed.value_counts().sort_index())
    print(f"Accuracy:                 {np.mean(accs_comb):.3f} ± {np.std(accs_comb):.3f}")
    print(f"Macro-F1 (all classes):   {np.mean(f1_macro_comb):.3f} ± {np.std(f1_macro_comb):.3f}")
    print(f"Macro-F1 (active only):   {np.mean(f1_macro_active_comb):.3f} ± {np.std(f1_macro_active_comb):.3f}")
    print("\nPer-class report:\n", rep_comb)

    with open(os.path.join(save_model_path, "report_combined_unsmoothed.txt"), "w") as f:
        f.write(rep_comb)
    pd.DataFrame(cm_comb).to_csv(os.path.join(save_model_path, "cm_combined_unsmoothed.csv"))
    
    
    # Confusion matrix
    import matplotlib.pyplot as plt
    labels_sorted = np.sort(np.unique(yt_all))
    disp_labels   = ["idle" if c == -1 else str(c) for c in labels_sorted]
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm_comb / cm_comb.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title("Normalized Confusion Matrix (Combined unsmothed)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels_sorted)))
    ax.set_xticklabels(disp_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels_sorted)))
    ax.set_yticklabels(disp_labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(os.path.join(save_model_path, "confusion_matrix_combined_unsmoothed.png"), dpi=150)
    plt.close(fig)
    
    # ------------------------- 8. TEMPORAL SMOOTHING APPLIED TO COMBINED PREDICTIONS ------------------------- 
    
    # Apply per participant:
    # - Don't smooth stage A as Idle is the majority class and smoothing might discard some active tasks
    # - Smooth Stage-B probabilities with EMA, then argmax.
    # - Rebuild probablilities:
    #   - p(idle | x_t) = 1 - pA
    #   - p(task + c | x_t) = pA*pB_smooth  
    
    alpha_B = 0.6 # smoothing factor for Stabe B only
    
    P_all_smooth = pd.DataFrame(
        0.0,
        index=yt_all.index,
        columns=all_classes
    )
    
    
    for pid, idx in participant_sequences(train_df.loc[yt_all.index], sort_key="id"):
        idx = idx.sort_values()
        
        pA_seq = pA_all.loc[idx].values          
        pB_seq = pB_all.loc[idx].values 
        
        pB_seq_smooth = ema_probs(pB_seq, alpha=alpha_B)
        
        # Rebuild full probabilities for this participant
        P_pid = pd.DataFrame(
            0.0,
            index=idx,
            columns=all_classes
        )
        
        P_pid[IDLE] = 1.0 - pA_seq
        
        for j, c in enumerate(active_classes):
            P_pid[c] = pA_seq * pB_seq_smooth[:, j]
        
        # Renormalize (small numerical safety)
        P_pid = P_pid.div(P_pid.sum(axis=1), axis=0)
        
        P_all_smooth.loc[idx] = P_pid
        
    yp_all_smooth = P_all_smooth.idxmax(axis=1)

    rep_smooth = classification_report(yt_all, yp_all_smooth, digits=3)
    cm_smooth  = confusion_matrix(yt_all, yp_all_smooth, labels=all_classes)
    
    labels_sorted = np.sort(np.unique(yt_all))
    disp_labels   = ["idle" if c == -1 else str(c) for c in labels_sorted]
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm_smooth / cm_smooth.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title("Normalized Confusion Matrix (smoothing)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels_sorted)))
    ax.set_xticklabels(disp_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels_sorted)))
    ax.set_yticklabels(disp_labels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(os.path.join(save_model_path, "confusion_matrix_smoothing.png"), dpi=150)
    plt.close(fig)
    
    print("\n=== Combined + Temporal smoothing ===")
    print(rep_smooth)
    with open(os.path.join(save_model_path, "report_combined_smoothed.txt"), "w") as f:
        f.write(rep_smooth)
    pd.DataFrame(cm_smooth).to_csv(os.path.join(save_model_path, "cm_combined_smoothed.csv"))                            
                                                       