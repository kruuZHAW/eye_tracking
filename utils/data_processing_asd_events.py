"""
asd_metrics_processor.py

Feature extraction for ASD (HMI) events over a single time window.

Usage
-----
processor = ASDEventsMetricsProcessor(window_asd)

asd_generic   = processor.compute_generic_features()
track_feats   = processor.compute_track_position_features()
transfer_feats= processor.compute_transfer_features()
popup_feats   = processor.compute_popup_features()
clear_feats   = processor.compute_clearance_features()

# If you also have eye-tracking window:
gaze_hmi_feats = processor.compute_gaze_mouse_label_track_features(eye_df_window)

# Or everything concatenated:
all_feats = processor.compute_all_metrics(eye_df_window=eye_df_window)
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

ALL_EVENT_TYPES = [
    "mouse_position",
    "track_screen_position",
    "track_label_position",
    "popup",
    "transfer",
    "clearance",
    "distance_measurement",
    # "speed_vector",
    # "sep_tool",
    # "route_interaction",
    # "keyboard_shortcut",
    # "track_mark",
]

TRANSFER_TYPES = [
    "TRANSFER",
    "ASSUME",
    "FORCE_ASSUME",
    "RELEASE",
    "REJECT_TRANSFER",
    "REQUEST_TRANSFER",
    "CANCEL_TRANSFER",
    "ACTIVATE_NEXT_SECTOR",
    "FORCE_ACT",
    "DECONTROL",
    "TRANSFER_TO_NEXT_SECTOR",
    "FORCE_RELEASE",
    "ENABLE_AUTO_CONTROL",
    "TRANSFER_TO_ANY",
    "MANUAL_OUTBOUND",
    "MANUAL_INBOUND",
]

CLEARANCE_TYPES = [
    "cleared-flight-level",
    "cleared-speed",
    "direct-to",
    "heading",
    "route-clearance",
]


# ---------------------------------------------------------------------------
# Generic ASD features
# ---------------------------------------------------------------------------

def compute_generic_asd_features(window_asd: pd.DataFrame) -> pd.DataFrame:
    """
    Computes generic ASD features for the given time window:
      - participant_id, scenario_id
      - n_events_total
      - n_events_unique
      - events_per_ms
      - events_per_timestamp
      - event_type_entropy
      - per-event-type counts
    """
    if window_asd.empty:
        return pd.DataFrame([{
            # "participant_id": None,
            # "scenario_id": None,
            "n_events_total": 0,
            "n_events_unique": 0,
            "events_per_ms": 0.0,
            "events_per_timestamp": 0.0,
            "event_type_entropy": 0.0,
            **{f"event_{ev}_count": 0 for ev in ALL_EVENT_TYPES},
        }])

    df_generic = {}

    # df_generic["participant_id"] = window_asd["participant_id"].max()
    # df_generic["scenario_id"] = window_asd["scenario_id"].max()

    df_generic["n_events_total"] = len(window_asd)
    df_generic["n_events_unique"] = window_asd.event_name.nunique()

    t_min = window_asd["epoch_ms"].min()
    t_max = window_asd["epoch_ms"].max()
    duration_ms = max(t_max - t_min + 1, 1)

    df_generic["events_per_ms"] = len(window_asd) / duration_ms
    df_generic["events_per_timestamp"] = len(window_asd) / max(window_asd["epoch_ms"].nunique(), 1)

    # High entropy = varied actions → possibly complex tasks
    # Low entropy = repetitive UI (monitoring)
    probs = window_asd.event_name.value_counts(normalize=True)
    df_generic["event_type_entropy"] = float(entropy(probs.values)) if not probs.empty else 0.0

    present_types = set(window_asd.event_name.unique())
    for ev in ALL_EVENT_TYPES:
        df_generic[f"event_{ev}_count"] = int((window_asd.event_name == ev).sum())

    return pd.DataFrame([df_generic])


# ---------------------------------------------------------------------------
# Flight lifecycle features (screen / label)
# ---------------------------------------------------------------------------

def compute_flight_lifecycle_features(
    df_tracks: pd.DataFrame,
    flight_col: str,
    prefix: str,
) -> Dict[str, int]:
    """
    Compute number of flights that appear, disappear, and persist
    over the course of the window, based on <flight_col>.

      prefix: used in feature names, e.g. "screen" → n_flights_screen_ever
    """
    out = {
        f"n_flights_{prefix}_ever": 0,
        f"n_flights_{prefix}_appear": 0,
        f"n_flights_{prefix}_disappear": 0,
        f"n_flights_{prefix}_persist": 0,
        f"n_flights_{prefix}_transient": 0,
    }

    if df_tracks.empty or flight_col not in df_tracks.columns:
        return out

    df_tracks = df_tracks.dropna(subset=[flight_col])
    if df_tracks.empty:
        return out

    # Set of flights for each epoch
    flights_by_epoch = (
        df_tracks
        .groupby("epoch_ms")[flight_col]
        .apply(lambda s: set(s))
    )

    if flights_by_epoch.empty:
        return out

    first_flights = flights_by_epoch.iloc[0]
    last_flights = flights_by_epoch.iloc[-1]

    # All flights seen at least once in the window
    flights_ever = set().union(*flights_by_epoch.tolist())

    # Appear: seen sometime, but not in the first timestamp
    flights_appear = flights_ever - first_flights

    # Disappear: seen sometime, but not in the last timestamp
    flights_disappear = flights_ever - last_flights

    # Persist: present both at start and end
    flights_persist = first_flights & last_flights

    # Transient: appear/disappear inside the window (not present both at start and end)
    flights_transient = flights_ever - (first_flights | last_flights)

    out[f"n_flights_{prefix}_ever"] = len(flights_ever)
    out[f"n_flights_{prefix}_appear"] = len(flights_appear)
    out[f"n_flights_{prefix}_disappear"] = len(flights_disappear)
    out[f"n_flights_{prefix}_persist"] = len(flights_persist)
    out[f"n_flights_{prefix}_transient"] = len(flights_transient)

    return out


# ---------------------------------------------------------------------------
# Track & label spatial / lifecycle features
# ---------------------------------------------------------------------------

def compute_track_position_features(window_asd: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate features for track_screen_position and track_label_position
    events within a single time window.

    Returns a 1-row DataFrame with:
      - participant_id, scenario_id
      - lifecycle stats for screen tracks and labels
      - counts and ratios for visibility / hovered / selected / on_pip
      - spatial stats (x/y, width/height, area)
    """
    features: Dict[str, float] = {}

    if window_asd.empty:
        return pd.DataFrame([features])

    # features["participant_id"] = window_asd["participant_id"].max()
    # features["scenario_id"] = window_asd["scenario_id"].max()

    # --- TRACK SCREEN POSITION FEATURES ---
    scr = window_asd[window_asd["event_name"] == "track_screen_position"].copy()

    lifecycle_screen = compute_flight_lifecycle_features(
        scr,
        flight_col="track_screen_position_flight_track_number",
        prefix="screen",
    )
    features.update(lifecycle_screen)

    if "track_screen_position_visible" in scr.columns:
        vis = scr["track_screen_position_visible"]
        features["track_screen_n_visible"] = int((vis == 1).sum())
        features["track_screen_visible_ratio"] = (
            features["track_screen_n_visible"] / max(len(scr), 1)
        )
    else:
        features["track_screen_n_visible"] = 0
        features["track_screen_visible_ratio"] = 0.0

    # Spatial stats for track_screen (x, y)
    for axis in ["x", "y"]:
        col = f"track_screen_position_{axis}"
        if col in scr.columns:
            vals = scr[col].dropna()
            features[f"track_screen_{axis}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            features[f"track_screen_{axis}_std"] = float(vals.std()) if len(vals) > 0 else np.nan
            features[f"track_screen_{axis}_min"] = float(vals.min()) if len(vals) > 0 else np.nan
            features[f"track_screen_{axis}_max"] = float(vals.max()) if len(vals) > 0 else np.nan
        else:
            features[f"track_screen_{axis}_mean"] = np.nan
            features[f"track_screen_{axis}_std"] = np.nan
            features[f"track_screen_{axis}_min"] = np.nan
            features[f"track_screen_{axis}_max"] = np.nan

    # --- TRACK LABEL POSITION FEATURES ---
    lab = window_asd[window_asd["event_name"] == "track_label_position"].copy()

    lifecycle_label = compute_flight_lifecycle_features(
        lab,
        flight_col="track_label_position_flight_track_number",
        prefix="label",
    )
    features.update(lifecycle_label)

    def _count_bool(col_name: str) -> int:
        if col_name in lab.columns:
            return int((lab[col_name] == 1).sum())
        return 0

    features["track_label_n_visible"] = _count_bool("track_label_position_visible")
    features["track_label_n_hovered"] = _count_bool("track_label_position_hovered")
    features["track_label_n_selected"] = _count_bool("track_label_position_selected")
    features["track_label_n_on_pip"] = _count_bool("track_label_position_on_pip")

    vis_count = max(features["track_label_n_visible"], 1)

    features["track_label_hovered_ratio"] = features["track_label_n_hovered"] / vis_count
    features["track_label_selected_ratio"] = features["track_label_n_selected"] / vis_count
    features["track_label_on_pip_ratio"] = features["track_label_n_on_pip"] / vis_count

    # Spatial stats for labels (x, y, width, height, area)
    for axis in ["x", "y"]:
        col = f"track_label_position_{axis}"
        if col in lab.columns:
            vals = lab[col].dropna()
            features[f"track_label_{axis}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            features[f"track_label_{axis}_std"] = float(vals.std()) if len(vals) > 0 else np.nan
            features[f"track_label_{axis}_min"] = float(vals.min()) if len(vals) > 0 else np.nan
            features[f"track_label_{axis}_max"] = float(vals.max()) if len(vals) > 0 else np.nan
        else:
            features[f"track_label_{axis}_mean"] = np.nan
            features[f"track_label_{axis}_std"] = np.nan
            features[f"track_label_{axis}_min"] = np.nan
            features[f"track_label_{axis}_max"] = np.nan

    # Width / height / area
    if {"track_label_position_width", "track_label_position_height"}.issubset(lab.columns):
        w = lab["track_label_position_width"].astype(float)
        h = lab["track_label_position_height"].astype(float)
        area = w * h

        features["track_label_width_mean"] = float(w.dropna().mean()) if w.notna().any() else np.nan
        features["track_label_height_mean"] = float(h.dropna().mean()) if h.notna().any() else np.nan
        features["track_label_area_mean"] = (
            float(area.dropna().mean()) if area.notna().any() else np.nan
        )
        features["track_label_area_total"] = float(area.dropna().sum()) if area.notna().any() else 0.0
    else:
        features["track_label_width_mean"] = np.nan
        features["track_label_height_mean"] = np.nan
        features["track_label_area_mean"] = np.nan
        features["track_label_area_total"] = 0.0

    return pd.DataFrame([features])


# ---------------------------------------------------------------------------
# Transfer features
# ---------------------------------------------------------------------------

def compute_transfer_features_window(asd_window: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transfer events over a time window into a single feature row.

    For each transfer type T in TRANSFER_TYPES, compute:
      - transfer_type_<T>_count
      - transfer_type_<T>_present  (0/1)
    """
    df_t = asd_window[asd_window["event_name"] == "transfer"].copy()

    features: Dict[str, float] = {}
    if df_t.empty:
        for t in TRANSFER_TYPES:
            base = f"transfer_type_{t}"
            features[f"{base}_count"] = 0
            features[f"{base}_present"] = 0
        return pd.DataFrame([features])

    counts = df_t["transfer_type_name"].value_counts().to_dict()

    for t in TRANSFER_TYPES:
        base = f"transfer_type_{t}"
        c = counts.get(t, 0)
        features[f"{base}_count"] = int(c)
        features[f"{base}_present"] = 1 if c > 0 else 0

    return pd.DataFrame([features])


# ---------------------------------------------------------------------------
# Popup features
# ---------------------------------------------------------------------------

def compute_popup_durations(df_popup: pd.DataFrame) -> List[Tuple[str, float, float]]:
    """
    Compute durations for popup *windows*, where a window is identified by:
        (popup_name, popup_flight_track_number)

    Returns a list of tuples:
        (popup_name, popup_flight_track_number, duration_ms)
    """
    df_popup = df_popup.sort_values("epoch_ms")
    durations: List[Tuple[str, float, float]] = []

    # (popup_name, flight_track) -> open_time_ms
    open_time: Dict[Tuple[str, float], float] = {}

    for _, row in df_popup.iterrows():
        name = row["popup_name"]
        flight_track = row["popup_flight_track_number"]
        t = row["epoch_ms"]

        key = (name, flight_track)

        if row["popup_opened"] == 1:
            open_time[key] = t

        elif row["popup_opened"] == 0 and key in open_time:
            durations.append((name, flight_track, t - open_time[key]))
            del open_time[key]

    return durations


def compute_popup_features_window(asd_window: pd.DataFrame) -> pd.DataFrame:
    """
    Computes features related to popups:

      - n_popup_open
      - n_popup_close
      - popup_any
      - popup_overlap (any time with more than 2 popups open)
      - popup_dwell_total_ms / mean / max
      - popup_per_flight_mean / max
      - popup_revisit_count
      - popup_inter_time_mean/median/std

    Returns a 1-row dataframe.
    """
    df_p = asd_window[asd_window["event_name"] == "popup"].copy()
    features: Dict[str, float] = {}

    if df_p.empty:
        features["n_popup_open"] = 0
        features["n_popup_close"] = 0
        features["popup_any"] = 0
        features["popup_overlap"] = 0
        features["popup_dwell_total_ms"] = 0
        features["popup_dwell_mean_ms"] = 0
        features["popup_dwell_max_ms"] = 0
        features["popup_per_flight_mean"] = 0
        features["popup_per_flight_max"] = 0
        features["popup_revisit_count"] = 0
        features["popup_inter_time_mean_ms"] = 0.0
        features["popup_inter_time_median_ms"] = 0.0
        features["popup_inter_time_std_ms"] = 0.0
        return pd.DataFrame([features])

    df_p = df_p.sort_values(["epoch_ms"])

    # Basic counts
    features["n_popup_open"] = int((df_p["popup_opened"] == 1).sum())
    features["n_popup_close"] = int((df_p["popup_opened"] == 0).sum())
    features["popup_any"] = 1

    # Approximate popup overlap: cumulative open minus close
    # Replace close (0) by -1, open (1) by +1, then cumsum
    open_close = df_p["popup_opened"].map({0: -1, 1: 1})
    overlap = (open_close.cumsum() > 2).any()
    features["popup_overlap"] = int(overlap)

    # Dwell time
    durations = compute_popup_durations(df_p)
    if durations:
        total = sum(d for _, _, d in durations)
        features["popup_dwell_total_ms"] = float(total)
        features["popup_dwell_mean_ms"] = float(total / len(durations))
        features["popup_dwell_max_ms"] = float(max(d for _, _, d in durations))
    else:
        features["popup_dwell_total_ms"] = 0.0
        features["popup_dwell_mean_ms"] = 0.0
        features["popup_dwell_max_ms"] = 0.0

    # flight features (count only opens per flight)
    if "popup_flight_track_number" in df_p.columns:
        nb_popup_per_flight = (
            df_p[df_p["popup_opened"] == 1]
            .groupby("popup_flight_track_number")
            .size()
        )
        if len(nb_popup_per_flight) > 0:
            features["popup_per_flight_mean"] = float(nb_popup_per_flight.mean())
            features["popup_per_flight_max"] = int(nb_popup_per_flight.max())
        else:
            features["popup_per_flight_mean"] = 0.0
            features["popup_per_flight_max"] = 0
    else:
        features["popup_per_flight_mean"] = 0.0
        features["popup_per_flight_max"] = 0

    # Revisit count (per (name, flight))
    revisit_count = 0
    last_state: Dict[Tuple[str, float], int] = {}

    for _, row in df_p.iterrows():
        key = (row["popup_name"], row["popup_flight_track_number"])
        opened = int(row["popup_opened"])

        # revisit: previous state = closed (0), now opened (1)
        if key in last_state and last_state[key] == 0 and opened == 1:
            revisit_count += 1

        last_state[key] = opened

    features["popup_revisit_count"] = revisit_count

    # Inter-popup time
    deltas = df_p["epoch_ms"].diff().dropna()
    if len(deltas) > 0:
        features["popup_inter_time_mean_ms"] = float(deltas.mean())
        features["popup_inter_time_median_ms"] = float(deltas.median())
        features["popup_inter_time_std_ms"] = float(deltas.std())
    else:
        features["popup_inter_time_mean_ms"] = 0.0
        features["popup_inter_time_median_ms"] = 0.0
        features["popup_inter_time_std_ms"] = 0.0

    return pd.DataFrame([features])


# ---------------------------------------------------------------------------
# Clearance features
# ---------------------------------------------------------------------------

def compute_clearance_features_window(asd_window: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clearance events over a time window into a single feature row.

    Features:
      - clearance_count
      - clearance_unique_flights
      - clearance_any
      - per-type counts & presence
      - inter-event timing stats
      - clearance_max_per_flight / mean_per_flight
    """
    df = asd_window[asd_window["event_name"] == "clearance"].copy()
    features: Dict[str, float] = {}

    # No clearances in this window
    if df.empty:
        features["clearance_count"] = 0
        features["clearance_unique_flights"] = 0
        features["clearance_any"] = 0

        for ctype in CLEARANCE_TYPES:
            base = f"clearance_type_{ctype}"
            features[f"{base}_count"] = 0
            features[f"{base}_present"] = 0

        features["clearance_inter_event_mean_ms"] = 0.0
        features["clearance_inter_event_median_ms"] = 0.0
        features["clearance_inter_event_std_ms"] = 0.0
        features["clearance_max_per_flight"] = 0
        features["clearance_mean_per_flight"] = 0.0
        return pd.DataFrame([features])

    df = df.sort_values("epoch_ms").reset_index(drop=True)

    features["clearance_count"] = len(df)
    features["clearance_any"] = 1

    # Flight-centric: use flight_track_number if available, else track_number
    flight_col = "clearance_flight_track_number"
    if flight_col not in df.columns or df[flight_col].isna().all():
        flight_col = "clearance_track_number"
    features["clearance_unique_flights"] = int(df[flight_col].nunique())

    # Per-type counts
    type_counts = df["clearance_type"].value_counts().to_dict()
    for ctype in CLEARANCE_TYPES:
        base = f"clearance_type_{ctype}"
        c = type_counts.get(ctype, 0)
        features[f"{base}_count"] = int(c)
        features[f"{base}_present"] = 1 if c > 0 else 0

    # Timing features
    deltas = df["epoch_ms"].diff().dropna()
    if len(deltas) > 0:
        features["clearance_inter_event_mean_ms"] = float(deltas.mean())
        features["clearance_inter_event_median_ms"] = float(deltas.median())
        features["clearance_inter_event_std_ms"] = float(deltas.std())
    else:
        features["clearance_inter_event_mean_ms"] = 0.0
        features["clearance_inter_event_median_ms"] = 0.0
        features["clearance_inter_event_std_ms"] = 0.0

    # Clearances per flight distribution
    per_flight_counts = df.groupby(flight_col).size()
    features["clearance_max_per_flight"] = int(per_flight_counts.max())
    features["clearance_mean_per_flight"] = float(per_flight_counts.mean())

    return pd.DataFrame([features])


# ---------------------------------------------------------------------------
# Cross-modal gaze / mouse vs label / track features
# ---------------------------------------------------------------------------

def compute_gaze_mouse_label_track_features(
    eye_df_window: pd.DataFrame,
    asd_window: pd.DataFrame,
    radius_R: float = 100.0,
    label_dwell_threshold_ms: float = 200.0,
    gaze_res: Tuple[int, int] = (1920, 1080),   # (width, height) px
    asd_res: Tuple[int, int] = (3840, 2160),    # (width, height) px
    toolbar_height: int = 27,                   # px
) -> pd.DataFrame:
    """
    Compute gaze/mouse vs label/track features for a single time window.

    Returns a 1-row dataframe with:
      - frac_gaze_toolbar
      - time_on_labels_ms, frac_time_on_labels, n_labels_looked_at
      - mean_dist_gaze_label, mean_dist_gaze_track
      - gaze_inside_label_ratio
      - gaze_inside_selected_label_ratio
      - gaze_inside_hovered_label_ratio
      - mouse_inside_label_ratio
      - gaze_near_track_ratio
      - mouse_near_track_ratio
    """

    gaze_w, gaze_h = gaze_res
    asd_w, asd_h = asd_res

    if eye_df_window.empty:
        return pd.DataFrame([{
            "frac_gaze_toolbar": 0.0,
            "time_on_labels_ms": 0.0,
            "frac_time_on_labels": 0.0,
            "n_labels_looked_at": 0,
            "mean_dist_gaze_label": np.nan,
            "mean_dist_gaze_track": np.nan,
            "gaze_inside_label_ratio": 0.0,
            "gaze_inside_selected_label_ratio": 0.0,
            "gaze_inside_hovered_label_ratio": 0.0,
            "mouse_inside_label_ratio": 0.0,
            "gaze_near_track_ratio": 0.0,
            "mouse_near_track_ratio": 0.0,
        }])

    # ---- 1. Eye data: epoch_ms + scale to ASD resolution ----
    eye = eye_df_window.copy().rename(
        columns={
            "Gaze point X [DACS px]": "gaze_x",
            "Gaze point Y [DACS px]": "gaze_y",
        }
    )
    if "epoch_ms" not in eye.columns:
        raise ValueError("eye_df_window must contain 'epoch_ms'")

    eye = eye.sort_values("epoch_ms").reset_index(drop=True)

    eye["gaze_x_scaled"] = eye["gaze_x"] * (asd_w / gaze_w)
    eye["gaze_y_scaled"] = eye["gaze_y"] * (asd_h / gaze_h)

    valid_mask = eye["gaze_x_scaled"].notna() & eye["gaze_y_scaled"].notna()
    eye_valid = eye.loc[valid_mask].copy()

    # Polaris main area (below toolbar)
    x_min, y_min, x_max, y_max = 0, toolbar_height, asd_w, asd_h
    in_polaris = (
        (eye_valid["gaze_x_scaled"] >= x_min) & (eye_valid["gaze_x_scaled"] <= x_max) &
        (eye_valid["gaze_y_scaled"] >= y_min) & (eye_valid["gaze_y_scaled"] <= y_max)
    )
    eye_polaris = eye_valid.loc[in_polaris].copy()
    eye_toolbar = eye_valid.loc[~in_polaris].copy()

    n_valid = len(eye_valid)
    frac_gaze_toolbar = len(eye_toolbar) / max(len(eye_valid), 1)

    if len(eye_polaris) == 0:
        return pd.DataFrame([{
            "frac_gaze_toolbar": frac_gaze_toolbar,
            "time_on_labels_ms": 0.0,
            "frac_time_on_labels": 0.0,
            "n_labels_looked_at": 0,
            "mean_dist_gaze_label": np.nan,
            "mean_dist_gaze_track": np.nan,
            "gaze_inside_label_ratio": 0.0,
            "gaze_inside_selected_label_ratio": 0.0,
            "gaze_inside_hovered_label_ratio": 0.0,
            "mouse_inside_label_ratio": 0.0,
            "gaze_near_track_ratio": 0.0,
            "mouse_near_track_ratio": 0.0,
        }])

    eye_polaris = eye_polaris.sort_values("epoch_ms").reset_index(drop=True)
    ts = eye_polaris["epoch_ms"].values.astype(float)
    if len(ts) > 1:
        dt = np.diff(ts)
        last_dt = np.median(dt)
        dt = np.concatenate([dt, [last_dt]])
    else:
        dt = np.array([0.0])
    eye_polaris.loc[:, "dt_ms"] = dt
    total_time_ms = eye_polaris["dt_ms"].sum()

    # ---- 2. Label state over time ----
    labels_all = asd_window[asd_window["event_name"] == "track_label_position"].copy()

    geom_mask = (
        labels_all["track_label_position_x"].notna() &
        labels_all["track_label_position_y"].notna() &
        labels_all["track_label_position_width"].notna() &
        labels_all["track_label_position_height"].notna()
    )
    labels_all = labels_all[geom_mask].copy()

    if not labels_all.empty:
        x = labels_all["track_label_position_x"]
        y = labels_all["track_label_position_y"]
        w = labels_all["track_label_position_width"]
        h = labels_all["track_label_position_height"]
        overlap_mask = ~(
            (x + w < 0) | (x > asd_w) |
            (y + h < 0) | (y > asd_h)
        )
        labels_all = labels_all[overlap_mask].copy()

    label_state_epochs = np.array([], dtype="int64")
    label_state: Dict[int, Dict[str, np.ndarray]] = {}

    if not labels_all.empty:
        labels_all = labels_all.sort_values("epoch_ms")
        label_state_epochs = labels_all["epoch_ms"].unique().astype("int64")

        for ep in label_state_epochs:
            labs_ep = labels_all[labels_all["epoch_ms"] == ep]

            rects = labs_ep[[
                "track_label_position_x",
                "track_label_position_y",
                "track_label_position_width",
                "track_label_position_height",
            ]].values.astype(float)

            cx = rects[:, 0] + rects[:, 2] / 2.0
            cy = rects[:, 1] + rects[:, 3] / 2.0

            selected = labs_ep.get(
                "track_label_position_selected",
                pd.Series(False, index=labs_ep.index)
            ).fillna(0).astype(bool).values

            hovered = labs_ep.get(
                "track_label_position_hovered",
                pd.Series(False, index=labs_ep.index)
            ).fillna(0).astype(bool).values

            if "track_label_position_track_number" in labs_ep.columns:
                label_ids = labs_ep["track_label_position_track_number"].values
            else:
                label_ids = np.arange(len(labs_ep))

            label_state[int(ep)] = {
                "rects": rects,
                "centers": np.vstack([cx, cy]).T,
                "selected": selected,
                "hovered": hovered,
                "ids": label_ids,
            }

    def get_label_state_at(t_ms: int):
        if label_state_epochs.size == 0:
            return None
        idx = np.searchsorted(label_state_epochs, t_ms, side="right") - 1
        if idx < 0:
            return None
        ep = int(label_state_epochs[idx])
        return label_state.get(ep, None)

    # ---- 3. Track state over time ----
    tracks_all = asd_window[asd_window["event_name"] == "track_screen_position"].copy()
    if {"track_screen_position_x", "track_screen_position_y"}.issubset(tracks_all.columns):
        tracks_all = tracks_all.dropna(subset=["track_screen_position_x", "track_screen_position_y"]).copy()
    else:
        tracks_all = tracks_all.iloc[0:0].copy()

    if not tracks_all.empty:
        tx = tracks_all["track_screen_position_x"]
        ty = tracks_all["track_screen_position_y"]
        on_screen = (
            (tx >= -radius_R) & (tx <= asd_w + radius_R) &
            (ty >= -radius_R) & (ty <= asd_h + radius_R)
        )
        tracks_all = tracks_all[on_screen].copy()

    track_state_epochs = np.array([], dtype="int64")
    track_state: Dict[int, Dict[str, np.ndarray]] = {}

    if not tracks_all.empty:
        tracks_all = tracks_all.sort_values("epoch_ms")
        track_state_epochs = tracks_all["epoch_ms"].unique().astype("int64")

        for ep in track_state_epochs:
            tr_ep = tracks_all[tracks_all["epoch_ms"] == ep]
            pts = tr_ep[["track_screen_position_x", "track_screen_position_y"]].values.astype(float)
            track_state[int(ep)] = {"points": pts}

    def get_track_state_at(t_ms: int):
        if track_state_epochs.size == 0:
            return None
        idx = np.searchsorted(track_state_epochs, t_ms, side="right") - 1
        if idx < 0:
            return None
        ep = int(track_state_epochs[idx])
        return track_state.get(ep, None)

    # ---- 4. Mouse events ----
    mouse_all = asd_window[asd_window["event_name"] == "mouse_position"].copy()
    if {"mouse_position_x", "mouse_position_y"}.issubset(mouse_all.columns):
        mouse_all = mouse_all.dropna(subset=["mouse_position_x", "mouse_position_y"]).copy()
    else:
        mouse_all = mouse_all.iloc[0:0].copy()

    if not mouse_all.empty:
        mx = mouse_all["mouse_position_x"]
        my = mouse_all["mouse_position_y"]
        in_screen = (
            (mx >= 0) & (mx <= asd_w) &
            (my >= 0) & (my <= asd_h)
        )
        mouse_all = mouse_all[in_screen].copy()

    mouse_all = mouse_all.sort_values("epoch_ms").reset_index(drop=True)

    # ---- 5. Iterate over gaze samples ----
    time_on_labels_ms = 0.0
    labels_dwell_ms: Dict[float, float] = defaultdict(float)

    gaze_inside_label_count = 0
    gaze_inside_selected_count = 0
    gaze_inside_hovered_count = 0
    gaze_near_track_count = 0

    gaze_min_dist_label: List[float] = []
    gaze_min_dist_track: List[float] = []

    gaze_x = eye_polaris["gaze_x_scaled"].values.astype(float)
    gaze_y = eye_polaris["gaze_y_scaled"].values.astype(float)
    gaze_ts = eye_polaris["epoch_ms"].values.astype("int64")
    dt_ms = eye_polaris["dt_ms"].values.astype(float)

    for gx, gy, t_ms, dt in zip(gaze_x, gaze_y, gaze_ts, dt_ms):
        lab_state = get_label_state_at(int(t_ms))
        inside_any_label = False
        inside_selected_label = False
        inside_hovered_label = False

        # Labels
        if lab_state is not None:
            rects = lab_state["rects"]
            centers = lab_state["centers"]
            selected = lab_state["selected"]
            hovered = lab_state["hovered"]
            ids = lab_state["ids"]

            dx = centers[:, 0] - gx
            dy = centers[:, 1] - gy
            dists = np.sqrt(dx * dx + dy * dy)
            gaze_min_dist_label.append(float(dists.min()))

            x0 = rects[:, 0]
            y0 = rects[:, 1]
            w = rects[:, 2]
            h = rects[:, 3]
            inside = (gx >= x0) & (gx <= x0 + w) & (gy >= y0) & (gy <= y0 + h)

            if inside.any():
                inside_any_label = True
                for flag, lbl_id in zip(inside, ids):
                    if flag:
                        labels_dwell_ms[lbl_id] += dt

                inside_selected_label = (inside & selected).any()
                inside_hovered_label = (inside & hovered).any()
        else:
            gaze_min_dist_label.append(np.nan)

        # Tracks
        tr_state = get_track_state_at(int(t_ms))
        if tr_state is not None:
            pts = tr_state["points"]
            dx_t = pts[:, 0] - gx
            dy_t = pts[:, 1] - gy
            dists_t = np.sqrt(dx_t * dx_t + dy_t * dy_t)
            min_dist_t = float(dists_t.min())
            gaze_min_dist_track.append(min_dist_t)
            if min_dist_t <= radius_R:
                gaze_near_track_count += 1
        else:
            gaze_min_dist_track.append(np.nan)

        # Aggregate per timestamp
        if inside_any_label:
            time_on_labels_ms += dt
            gaze_inside_label_count += 1
        if inside_selected_label:
            gaze_inside_selected_count += 1
        if inside_hovered_label:
            gaze_inside_hovered_count += 1

    # ---- 6. Mouse-based features ----
    mouse_inside_label_count = 0
    mouse_near_track_count = 0
    n_mouse = len(mouse_all)

    for _, row in mouse_all.iterrows():
        mx = float(row["mouse_position_x"])
        my = float(row["mouse_position_y"])
        t_ms = int(row["epoch_ms"])

        lab_state = get_label_state_at(t_ms)
        if lab_state is not None:
            rects = lab_state["rects"]
            x0 = rects[:, 0]
            y0 = rects[:, 1]
            w = rects[:, 2]
            h = rects[:, 3]
            inside = (mx >= x0) & (mx <= x0 + w) & (my >= y0) & (my <= y0 + h)
            if inside.any():
                mouse_inside_label_count += 1

        tr_state = get_track_state_at(t_ms)
        if tr_state is not None:
            pts = tr_state["points"]
            dx_t = pts[:, 0] - mx
            dy_t = pts[:, 1] - my
            dists_t = np.sqrt(dx_t * dx_t + dy_t * dy_t)
            if dists_t.min() <= radius_R:
                mouse_near_track_count += 1

    # ---- 7. Aggregate final features ----
    time_on_labels_ms = float(time_on_labels_ms)
    frac_time_on_labels = float(time_on_labels_ms / total_time_ms) if total_time_ms > 0 else 0.0

    n_labels_looked_at = sum(
        dwell >= label_dwell_threshold_ms for dwell in labels_dwell_ms.values()
    )

    mean_dist_gaze_label = (
        float(np.nanmean(gaze_min_dist_label)) if len(gaze_min_dist_label) > 0 else np.nan
    )
    mean_dist_gaze_track = (
        float(np.nanmean(gaze_min_dist_track)) if len(gaze_min_dist_track) > 0 else np.nan
    )

    gaze_inside_label_ratio = (
        gaze_inside_label_count / n_valid if n_valid > 0 else 0.0
    )
    gaze_inside_selected_ratio = (
        gaze_inside_selected_count / n_valid if n_valid > 0 else 0.0
    )
    gaze_inside_hovered_ratio = (
        gaze_inside_hovered_count / n_valid if n_valid > 0 else 0.0
    )
    mouse_inside_label_ratio = (
        mouse_inside_label_count / n_mouse if n_mouse > 0 else 0.0
    )
    gaze_near_track_ratio = (
        gaze_near_track_count / n_valid if n_valid > 0 else 0.0
    )
    mouse_near_track_ratio = (
        mouse_near_track_count / n_mouse if n_mouse > 0 else 0.0
    )

    out = {
        "frac_gaze_toolbar": frac_gaze_toolbar,
        "time_on_labels_ms": time_on_labels_ms,
        "frac_time_on_labels": frac_time_on_labels,
        "n_labels_looked_at": n_labels_looked_at,
        "mean_dist_gaze_label": mean_dist_gaze_label,
        "mean_dist_gaze_track": mean_dist_gaze_track,
        "gaze_inside_label_ratio": gaze_inside_label_ratio,
        "gaze_inside_selected_label_ratio": gaze_inside_selected_ratio,
        "gaze_inside_hovered_label_ratio": gaze_inside_hovered_ratio,
        "mouse_inside_label_ratio": mouse_inside_label_ratio,
        "gaze_near_track_ratio": gaze_near_track_ratio,
        "mouse_near_track_ratio": mouse_near_track_ratio,
    }

    return pd.DataFrame([out])


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------

class ASDEventsMetricsProcessor:
    """
    High-level wrapper around ASD event feature functions for a single time window.

    Similar spirit to MouseMetricsProcessor / GazeMetricsProcessor.
    """

    def __init__(self, window_asd: pd.DataFrame):
        self.asd = window_asd.sort_values("epoch_ms").reset_index(drop=True)

    # Individual groups
    def compute_generic_features(self) -> pd.DataFrame:
        return compute_generic_asd_features(self.asd)

    def compute_track_position_features(self) -> pd.DataFrame:
        return compute_track_position_features(self.asd)

    def compute_transfer_features(self) -> pd.DataFrame:
        return compute_transfer_features_window(self.asd)

    def compute_popup_features(self) -> pd.DataFrame:
        return compute_popup_features_window(self.asd)

    def compute_clearance_features(self) -> pd.DataFrame:
        return compute_clearance_features_window(self.asd)

    def compute_gaze_mouse_label_track_features(
        self,
        eye_df_window: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        return compute_gaze_mouse_label_track_features(
            eye_df_window=eye_df_window,
            asd_window=self.asd,
            **kwargs,
        )

    def compute_all_metrics(
        self,
        eye_df_window: Optional[pd.DataFrame] = None,
        **gaze_kwargs,
    ) -> pd.DataFrame:
        """
        Compute all ASD-only metrics, and optionally cross-modal gaze/mouse vs HMI metrics
        if `eye_df_window` is provided.

        Returns a single 1-row DataFrame with all columns concatenated.
        """
        dfs = [
            self.compute_generic_features(),
            self.compute_track_position_features(),
            self.compute_transfer_features(),
            self.compute_popup_features(),
            self.compute_clearance_features(),
        ]

        if eye_df_window is not None:
            dfs.append(
                self.compute_gaze_mouse_label_track_features(
                    eye_df_window=eye_df_window,
                    **gaze_kwargs,
                )
            )

        return pd.concat(dfs, axis=1)
