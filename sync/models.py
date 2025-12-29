from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class ClockPointState(Enum):
    PLACEHOLDER = "PLACEHOLDER" # Initial boundaries (Unverified)
    PREDICTED = "PREDICTED" # Interpolated (Unverified)
    CONFIRMED = "CONFIRMED" # User verified

@dataclass
class ClockPoint:
    """Represents a timestamp on a specific machine."""
    clock_time: datetime
    frame: int
    pts_ms: float | None
    state: ClockPointState

    @property
    def timestamp(self) -> float:
        return self.clock_time.timestamp() * 1000.0
    
    @staticmethod
    def from_dict(data: dict) -> "ClockPoint":
        return ClockPoint(
            clock_time=datetime.fromisoformat(data["clock_iso"]),
            frame=data["frame"],
            pts_ms=data["pts_ms"],
            state=ClockPointState(data["state"])
        )

@dataclass
class SyncPair:
    """Two nearest timestamps of recorder and participant for synchronization at a given time."""
    recorder: ClockPoint
    participant: ClockPoint

    @property
    def offset(self) -> float:
        clock_diff = self.recorder.timestamp - self.participant.timestamp
        video_diff = self.recorder.pts_ms - self.participant.pts_ms
        return clock_diff - video_diff

    @property
    def is_confirmed(self) -> bool:
        return (self.recorder.state == ClockPointState.CONFIRMED and 
                self.participant.state == ClockPointState.CONFIRMED)
    
    @staticmethod
    def from_dict(data: dict) -> "SyncPair":
        return SyncPair(
            recorder=ClockPoint.from_dict(data["recorder"]),
            participant=ClockPoint.from_dict(data["participant"])
        )