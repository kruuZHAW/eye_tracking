import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

from sync.models import SyncPair

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler().setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")))

class TimeSynchronizer:
    # Variables for interpolation
    xp: np.ndarray
    """X-coordinates (Recorder timestamp)"""
    fp: np.ndarray
    """Y-coordinates (Offset)"""

    # Constants
    WARNING_LOW_COVERAGE_PERCENT: float = 0.85
    """Percentage of data expected to be covered between interpolation boundaries."""
    WARNING_TIME_DELTA_FROM_BOUNDARY_MS: int = 5 * 60 * 1000
    """Max expected time delta from boundaries for extrapolation (in ms)."""

    def __init__(self, path: Path):
        """
        Initialize TimeSynchronizer with synchronization points from JSON file.
        
        :param path: Path to JSON file containing synchronization points.
        :type path: Path
        """
        with open(path, 'r') as f:
            data: dict = json.load(f)

        # Validate sync points is not empty or missing
        if not (sync_points_data := data.get("sync_points", [])):
            raise ValueError(f"No 'sync_points' found in {path}.")
        
        # Build SyncPais, filter by state and sort by recorder time
        sync_points: list[SyncPair] = [SyncPair.from_dict(data) for data in sync_points_data]
        sync_points = sorted(
            filter(lambda p: p.is_confirmed, sync_points),
            key=lambda p: p.recorder.timestamp
        )

        # Validate existance of confirmed sync points
        if (count := len(sync_points)) == 0:
            raise ValueError(f"No confirmed synchronization points found in {path}.")

        # Build arguments for interpolation
        self.xp = np.fromiter((p.recorder.timestamp for p in sync_points), dtype=np.float64, count=count)
        self.fp = np.fromiter((p.offset for p in sync_points), dtype=np.float64, count=count)

    def __call__(self, df: pd.DataFrame, timezone: str = "Europe/Zagreb") -> np.ndarray:
        """
        Use synchronization points to convert eye tracking data timestamps to Polaris participant clock.
        
        :param df: DataFrame of eye tracking data from TSV file.
        :type df: pd.DataFrame
        :param timezone: Timezone of computer recording eye tracking data. Defaults to "Europe/Zagreb".
        :type timezone: str
        :return: Array of synchronized timestamps in milliseconds.
        :rtype: ndarray
        """

        # Parse start datetime, subtract relative ts (should be 0, but just to be safe) and get UTC timestamp in ms
        start_event = df[df["Event"].eq("RecordingStart")].iloc[0]
        
        # Parse date and time to datetime with timezone
        start_dt = datetime.strptime(
            f"{start_event['Recording date']} {start_event['Recording start time']}",
            "%d.%m.%Y. %H:%M:%S.%f"
        ).replace(tzinfo=ZoneInfo(timezone)) - timedelta(milliseconds=int(start_event["Recording timestamp [ms]"]))
        
        # Convert start datetime to UTC ms timestamp
        start_ms = start_dt.timestamp() * 1000
        
        # Convert relative timestamp to UTC timestamp as float64 for interpolation precision
        ts = np.asarray(df["Recording timestamp [ms]"] + start_ms, dtype=np.float64)

        # Constant offset
        if len(self.xp) == 1:
            return ts - self.fp[0]
        
        # Elements outside sync boundaries
        mask_right = ts > self.xp[-1]
        mask_left = ts < self.xp[0]
        
        # Statistics
        n_total = len(ts)
        n_left = np.count_nonzero(mask_left)
        n_right = np.count_nonzero(mask_right)
        n_inside = n_total - n_left - n_right

        logger.debug(
            f"Sync Coverage Breakdown: "
            f"Left Extrap={n_left:,} ({n_left/n_total:.1%}) | "
            f"Inside={n_inside:,} ({n_inside/n_total:.1%}) | "
            f"Right Extrap={n_right:,} ({n_right/n_total:.1%})"
        )

        if (coverage := n_inside / n_total) < self.WARNING_LOW_COVERAGE_PERCENT:
            logger.warning(
                f"Low Sync Coverage: Only {coverage:.2%} of data points are within the verified sync boundaries."
            )

        # Main interpolation
        offsets = np.interp(ts, self.xp, self.fp)

        # Linear extrapolation before first sync_point
        if np.any(mask_left):
            slope = (self.fp[1] - self.fp[0]) / (self.xp[1] - self.xp[0])
            offsets[mask_left] = self.fp[0] + slope * (ts[mask_left] - self.xp[0])

            # Check time delta
            if (time_delta := self.xp[0] - np.min(ts[mask_left])) > self.WARNING_TIME_DELTA_FROM_BOUNDARY_MS:
                logger.warning(
                    f"Excessive Left Extrapolation: Time delta of {time_delta / 60_000:.0f} minutes "
                    "between first timestamp and first sync point."
                )

        # Linear extrapolation after last sync_point
        if np.any(mask_right):
            slope = (self.fp[-1] - self.fp[-2]) / (self.xp[-1] - self.xp[-2])
            offsets[mask_right] = self.fp[-1] + slope * (ts[mask_right] - self.xp[-1])

            # Check time delta
            if (time_delta := np.max(ts[mask_right]) - self.xp[-1]) > self.WARNING_TIME_DELTA_FROM_BOUNDARY_MS:
                logger.warning(
                    f"Excessive Right Extrapolation: Time delta of {time_delta / 60_000:.0f} minutes between last "
                    "sync point and last timestamp."
                )

        # Apply offsets
        return ts - offsets