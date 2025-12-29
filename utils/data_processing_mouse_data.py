import pandas as pd
import numpy as np

# TODO: Maybe modify IDLE time as the time between consecutive mouse positions and they are transmitted only when the mouse is moving

class MouseMetricsProcessor:
    """
    Computes mouse movement-related metrics from a DataFrame with a single task execution.
    Automatically handles variations in column names (with/without units) and timestamp units.
    """

    def __init__(self, df: pd.DataFrame, resample = False):
        self.original_df = df.copy()
        

        self.ts_col = self._find_col(df, "epoch_ms")
        self.mx_col = self._find_col(df, "mouse_position_x")
        self.my_col = self._find_col(df, "mouse_position_y")

        self.df = df.sort_values(by=self.ts_col).reset_index(drop=True).copy()
        
        self._collapse_duplicate_mouse_timestamps()
        
        if resample:
            self._resample_fixed_timestep(step_ms=20) # 50 Hz

    def _find_col(self, df: pd.DataFrame, prefix: str):
        for col in df.columns:
            if col.startswith(prefix):
                return col
        raise ValueError(f"Required column starting with '{prefix}' not found.")
    
    def _resample_fixed_timestep(self, step_ms=20):
        """
        Resample mouse data to fixed timestep by forward-filling positions.
        """
        df = self.original_df.sort_values(self.ts_col).copy()

        if df.empty:
            return

        t0, t1 = df[self.ts_col].iloc[0], df[self.ts_col].iloc[-1]
        if t1 <= t0:
            return

        t_grid = np.arange(t0, t1 + step_ms, step_ms)

        df_resampled = (
            df.set_index(self.ts_col)
              .reindex(t_grid)
              .ffill()
              .reset_index()
              .rename(columns={"index": self.ts_col})
        )

        self.df = df_resampled
        
    def _collapse_duplicate_mouse_timestamps(self):
        """
        Handle mouse positions transmitted at the same timestamp
        by collapsing rows with identical timestamps.

        - mouse_position_x / mouse_position_y: mean
        - all other columns: first (you can change to 'last' if you prefer)
        """
        df = self.df.sort_values(self.ts_col)

        agg_dict = {
            self.mx_col: "mean",
            self.my_col: "mean",
        }
        other_cols = [c for c in df.columns if c not in [self.ts_col, self.mx_col, self.my_col]]
        for c in other_cols:
            agg_dict[c] = "first"

        df_collapsed = (
            df.groupby(self.ts_col, as_index=False)
              .agg(agg_dict)
        )

        self.df = df_collapsed

    def _to_seconds(self, s: pd.Series) -> pd.Series:
        return s * 1e-3 # always in ms

    def _time_diff_seconds(self) -> pd.Series:
        return self._to_seconds(self.df[self.ts_col].diff().fillna(0))

    # ------------------------- VELOCITY & ACCELERATION -------------------------
    def compute_velocity_acceleration(self):
        dx = self.df[self.mx_col].diff().fillna(0)
        dy = self.df[self.my_col].diff().fillna(0)
        disp = np.sqrt(dx**2 + dy**2)
        dt = self._time_diff_seconds().replace(0, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'): # Avoid warnings if nan or 0 dt
            velocity = disp / dt
            acceleration = velocity.diff().fillna(0) / dt
        return velocity, acceleration

    # ------------------------- MOVEMENT FREQUENCY & IDLE TIMES -------------------------
    def compute_movement_frequency_idle_time(self, idle_threshold=0.5):
        dt = self._time_diff_seconds()

        total_time = dt.sum()
        if total_time <= 0 or np.isnan(total_time):
            return 0.0, 0.0

        moved = (self.df[self.mx_col].diff().abs() > 0) | \
                (self.df[self.my_col].diff().abs() > 0)

        # Safe division (protect against divide-by-zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            movement_frequency = moved.sum() / total_time

        idle_periods = dt[~moved]

        # sum only valid idle periods
        idle_periods = idle_periods[idle_periods >= idle_threshold]

        total_idle_time = idle_periods.sum()
        if np.isnan(total_idle_time):
            total_idle_time = 0.0

        return float(movement_frequency), float(total_idle_time)

    # ------------------------- DIRECTION CHANGES -------------------------
    def compute_path_patterns(self, angle_threshold=30):
        dx = self.df[self.mx_col].diff().fillna(0)
        dy = self.df[self.my_col].diff().fillna(0)
        angles = np.arctan2(dy, dx)
        angle_diff = np.abs(np.diff(angles))
        changes = np.sum(angle_diff > np.radians(angle_threshold))
        return int(changes)

    # ------------------------- DISTANCE & STOPS -------------------------
    def compute_total_distance_and_stops(self, stop_threshold=5):
        dx = self.df[self.mx_col].diff().fillna(0)
        dy = self.df[self.my_col].diff().fillna(0)
        disp = np.sqrt(dx**2 + dy**2)
        dt = self._time_diff_seconds().replace(0, np.nan)
        velocity = disp / dt
        total_distance = disp.sum()
        num_stops = (velocity < stop_threshold).sum()
        return total_distance, int(num_stops)

    # ------------------------- MOVEMENT BURSTS & STILLNESS -------------------------
    def compute_movement_bursts(self, burst_threshold=50, stillness_threshold=5):
        dx = self.df[self.mx_col].diff().fillna(0)
        dy = self.df[self.my_col].diff().fillna(0)
        disp = np.sqrt(dx**2 + dy**2)
        dt = self._time_diff_seconds().replace(0, np.nan)
        velocity = disp / dt
        bursts = (velocity > burst_threshold).sum()
        stillness = (velocity < stillness_threshold).sum()
        return int(bursts), int(stillness)

    # ------------------------- AGGREGATE -------------------------
    def compute_all_metrics(self):
        velocity, acceleration = self.compute_velocity_acceleration()
        movement_freq, idle_time = self.compute_movement_frequency_idle_time()
        direction_changes = self.compute_path_patterns()
        total_distance, num_stops = self.compute_total_distance_and_stops()
        bursts, stillness = self.compute_movement_bursts()
        
        if self.original_df.empty:
            return {
                "Avg Mouse Velocity (px/s)": 0,
                "Avg Mouse Acceleration (px/s²)": 0,
                "Movement Frequency (movements/s)": 0,
                "Total Idle Time (s)": 0,
                "Path Direction Changes": 0,
                "Total Distance Traveled (px)": 0,
                "Number of Stops": 0,
                "Movement Bursts": 0,
                "Stillness Periods": 0,
            }

        return {
            "Avg Mouse Velocity (px/s)": velocity.mean(),
            "Avg Mouse Acceleration (px/s²)": acceleration.mean(),
            "Movement Frequency (movements/s)": movement_freq,
            "Total Idle Time (s)": idle_time,
            "Path Direction Changes": direction_changes,
            "Total Distance Traveled (px)": total_distance,
            "Number of Stops": num_stops,
            "Movement Bursts": bursts,
            "Stillness Periods": stillness,
            # "Seconds per raw time unit": self.seconds_per_unit,
            # "Timestamp column": self.ts_col,
            # "Mouse X column": self.mx_col,
            # "Mouse Y column": self.my_col,
        }