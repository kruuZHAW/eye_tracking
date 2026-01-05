import re
import shutil
from pathlib import Path

PATTERN = re.compile(
    r"^(?P<pid>\d{3})_scenario_(?P<sid>\d+)_gaze_data_fusion\.tsv$",
    re.IGNORECASE
)

def materialize(flat_dir: Path, dest_root: Path, mode: str = "symlink"):
    """
    flat_dir: folder containing reviewed TSVs (by Jenny) in flat structure
    dest_root: root of old-style tree (pid/Scenario X/ET)
    mode: 'symlink', 'copy'
    """
    flat_dir = flat_dir.resolve()
    dest_root = dest_root.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    n = 0
    for tsv in flat_dir.glob("*.tsv"):
        m = PATTERN.match(tsv.name)
        if not m:
            continue
        pid = m.group("pid")
        sid = m.group("sid")

        et_dir = dest_root / pid / f"Scenario {sid}" / "ET"
        et_dir.mkdir(parents=True, exist_ok=True)

        out_name = tsv.stem + "_reviewed.tsv"
        out = et_dir / out_name
        if out.exists():
            continue

        if mode == "symlink":
            out.symlink_to(tsv)
        elif mode == "copy":
            shutil.copy2(tsv, out)
        else:
            raise ValueError("mode must be 'symlink' or 'copy'")

        n += 1

    print(f"[ok] Materialized {n} ET files into {dest_root}")

if __name__ == "__main__":
    import sys
    flat = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else "symlink"
    materialize(flat, dest, mode)
