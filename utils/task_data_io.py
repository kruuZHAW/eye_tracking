import os, re
from pathlib import Path
import pandas as pd
from typing import Union, Iterable, Optional

PARQUET_NAME = "raw_inputs.parquet"

def _parse_ids_from_path(p: Path) -> tuple[str, str]:
    """
    Expect: <root>/<participant_id>/Scenario <scenario_id>/taskRecognition/raw_inputs.parquet
    """
    # p ... / taskRecognition / raw_inputs.parquet
    scenario_seg = p.parents[1].name       # 'Scenario 3'
    pid         = p.parents[2].name        # '010', etc.
    m = re.match(r"Scenario\s+(\d+)$", scenario_seg)
    if not m:
        raise ValueError(f"Cannot parse scenario id from {scenario_seg!r}")
    return pid, m.group(1)

def list_parquet_files(
    root: Union[str, Path],
    participants: Optional[Iterable[str]] = None,
    scenarios: Optional[Iterable[Union[str, int]]] = None,
) -> list[dict]:
    """
    Finds: <root>/<pid>/Scenario <sid>/taskRecognition/raw_inputs.parquet
    Returns [{'path': Path, 'participant_id': str, 'scenario_id': str}, ...]
    """
    root = Path(root)
    part_set = set(map(str, participants)) if participants else None
    scen_set = set(map(str, scenarios)) if scenarios else None

    out = []
    # search relative to root; no hard-coded 'training_data' string
    for p in root.glob("*/Scenario */taskRecognition/" + PARQUET_NAME):
        try:
            pid, sid = _parse_ids_from_path(p)
        except ValueError:
            continue
        if part_set and pid not in part_set:
            continue
        if scen_set and sid not in scen_set:
            continue
        out.append({"path": p, "participant_id": pid, "scenario_id": sid})

    out.sort(key=lambda d: (str(d["participant_id"]), int(d["scenario_id"]), str(d["path"])))
    return out