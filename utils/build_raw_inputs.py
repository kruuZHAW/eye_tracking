import os, sys, sqlite3, re, traceback
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# --- imports for protobuf mouse decoding ---
try:
    from gen import messages_pb2  # created by events/setup.sh
except Exception as e:
    print("ERROR: cannot import gen.messages_pb2 – did you run ./setup.sh and activate the venv?", file=sys.stderr)
    raise

TZ = ZoneInfo("Europe/Zagreb")

# ---------- helpers: discovery ----------
def find_scenarios(root: Path) -> List[Tuple[str, str, Path]]:
    """
    Return list of (participant_id, scenario_id, scenario_dir) under:
      root/{pid}/Scenario {sid}/
    """
    out = []
    for pid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        for scen_dir in sorted(pid_dir.glob("Scenario *")):
            if not scen_dir.is_dir(): continue
            m = re.match(r"Scenario\s+(\d+)$", scen_dir.name)
            if not m: continue
            sid = m.group(1)
            out.append((pid, sid, scen_dir))
    return out

def find_et_tsv(scen_dir: Path) -> Optional[Path]:
    et_dir = scen_dir / "ET"
    if not et_dir.is_dir(): return None
    # Prefer fusion TSVs if present, else any TSV
    cand = list(et_dir.glob("*gaze*fusion*.tsv"))
    if not cand:
        cand = list(et_dir.glob("*.tsv"))
    return sorted(cand)[0] if cand else None

def find_sim_db(scen_dir: Path) -> Optional[Path]:
    sim_dir = scen_dir / "simulator"
    if not sim_dir.is_dir(): return None
    # usually simulator/{pid}_scenario_{sid}/polaris-events-*.db
    cand = sorted(sim_dir.rglob("polaris-events-*.db"))
    if not cand: return None
    # pick the most recent by mtime
    return max(cand, key=lambda p: p.stat().st_mtime)

def taskrecognition_dir(scen_dir: Path) -> Path:
    d = scen_dir / "taskRecognition"
    d.mkdir(parents=True, exist_ok=True)
    return d

def needs_rebuild(tsv: Path, db: Path, out: Path) -> bool:
    if not out.exists(): return True
    src_mtime = max(tsv.stat().st_mtime, db.stat().st_mtime)
    return out.stat().st_mtime < src_mtime

# ---------- ET: load & slice, compute epoch_ms ----------
def slice_between_events(df: pd.DataFrame, start="ScreenRecordingStart", end="ScreenRecordingEnd", include_bounds=True) -> pd.DataFrame:
    s = df["Event"]
    starts = np.flatnonzero(s.eq(start))
    ends   = np.flatnonzero(s.eq(end))
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"Expected exactly one '{start}' and one '{end}' (got {len(starts)} and {len(ends)}).")
    i, j = int(starts[0]), int(ends[0])
    if j < i:
        raise ValueError(f"'{end}' occurs before '{start}'")
    return df.iloc[i:j+1] if include_bounds else df.iloc[i+1:j]

def build_et_frame(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    df = slice_between_events(df, include_bounds=True)

    # Clean up ET columns (drop ET mouse if present; we’ll use simulator mouse)
    for c in ["Mouse position X [DACS px]", "Mouse position Y [DACS px]"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # CET absolute timestamp then UNIX epoch (UTC ms)
    date_str = df["Recording date"].astype(str).str.replace(r"\.$", "", regex=True)
    base_start = pd.to_datetime(
        date_str + " " + df["Recording start time"].astype(str),
        format="%d.%m.%Y %H:%M:%S.%f",
        errors="coerce",
    )
    offset = pd.to_timedelta(pd.to_numeric(df["Recording timestamp [ms]"], errors="coerce"), unit="ms")
    ts_cet = base_start + offset
    ts_cet = ts_cet.dt.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
    df["epoch_ms"] = (ts_cet.view("int64") // 1_000_000).astype("int64")

    # keep only what we need from ET; can be adapted
    keep = [
        "epoch_ms", "Recording timestamp [ms]", "Event",
        "Gaze point X [DACS px]", "Gaze point Y [DACS px]",
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].sort_values("Recording timestamp [ms]")

# ---------- Simulator: load mouse for window ----------
def load_mouse_positions(db_path: Path, start_ms: int, end_ms: int, batch=50_000) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    con.text_factory = bytes
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA mmap_size=268435456")
    con.execute("PRAGMA temp_store=MEMORY")

    sql = ('SELECT id, epoch_ms, payload FROM "events" '
           'WHERE epoch_ms BETWEEN ? AND ? '
           'ORDER BY epoch_ms')
    cur = con.execute(sql, (int(start_ms), int(end_ms)))

    ids, epochs, xs, ys = [], [], [], []
    while True:
        rows = cur.fetchmany(batch)
        if not rows: break
        for id_, ms, blob in rows:
            ev = messages_pb2.Event()
            ev.ParseFromString(blob)
            mp = getattr(getattr(ev, "asd_event", None), "mouse_position", None)
            x = getattr(mp, "x", None) or getattr(mp, "pos_x", None) if mp is not None else None
            y = getattr(mp, "y", None) or getattr(mp, "pos_y", None) if mp is not None else None
            ids.append(int(id_)); epochs.append(int(ms)); xs.append(x); ys.append(y)
    con.close()

    df = pd.DataFrame({"epoch_ms": epochs, "Mouse position X": xs, "Mouse position Y": ys, "id": ids})
    return df.sort_values("epoch_ms")

# ---------- Merge & write ----------
def merge_and_write(et: pd.DataFrame, mouse: pd.DataFrame, out_parquet: Path, tol_ms=0):
    # Union timeline, then nearest-asof with tolerance
    dfe = et.astype({"epoch_ms":"int64"}).sort_values("epoch_ms")
    dfm = mouse.astype({"epoch_ms":"int64"}).sort_values("epoch_ms")[["epoch_ms","Mouse position X","Mouse position Y"]]

    timeline = pd.DataFrame({"epoch_ms": np.union1d(dfe["epoch_ms"].values, dfm["epoch_ms"].values)})

    merged = pd.merge_asof(timeline, dfe, on="epoch_ms", direction="nearest", tolerance=tol_ms)
    merged = pd.merge_asof(merged, dfm, on="epoch_ms", direction="nearest", tolerance=tol_ms)

    ts_utc = pd.to_datetime(merged["epoch_ms"], unit="ms", utc=True)
    merged["ts_cet"] = ts_utc.dt.tz_convert(TZ)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_parquet, index=False)
    return merged

# ---------- Orchestrate one scenario ----------
def process_scenario(scen_dir: Path) -> Optional[Path]:
    et = find_et_tsv(scen_dir)
    db = find_sim_db(scen_dir)
    if not et or not db:
        print(f"[skip] Missing ET or DB in: {scen_dir}", file=sys.stderr)
        return None

    out_parquet = taskrecognition_dir(scen_dir) / "raw_inputs.parquet"
    if not needs_rebuild(et, db, out_parquet):
        print(f"[ok] Up-to-date: {out_parquet}")
        return out_parquet

    try:
        df_et = build_et_frame(et)
        if df_et.empty:
            print(f"[warn] ET slice empty in {et}", file=sys.stderr)
            return None
        df_mouse = load_mouse_positions(db, int(df_et["epoch_ms"].min()), int(df_et["epoch_ms"].max()))
        merged = merge_and_write(df_et, df_mouse, out_parquet, tol_ms=0)
        print(f"[write] {out_parquet}  rows={len(merged)}")
        return out_parquet
    except Exception as e:
        print(f"[error] {scen_dir}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# ---------- CLI ----------
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /store/…/training_data [optional: pid sid]", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1])

    # Optional filters: participant_id and/or scenario_id
    pid_filter = sys.argv[2] if len(sys.argv) >= 3 else None
    sid_filter = sys.argv[3] if len(sys.argv) >= 4 else None

    scenarios = find_scenarios(root)
    if pid_filter:
        scenarios = [t for t in scenarios if t[0] == pid_filter]
    if sid_filter:
        scenarios = [t for t in scenarios if t[1] == sid_filter]

    if not scenarios:
        print("[warn] No scenarios found.", file=sys.stderr)

    wrote = 0
    for pid, sid, scen_dir in scenarios:
        res = process_scenario(scen_dir)
        if res: wrote += 1
    print(f"Done. Wrote/confirmed {wrote} parquet file(s).")

if __name__ == "__main__":
    main()
