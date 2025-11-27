import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Iterable, Tuple
from google.protobuf.json_format import MessageToDict
from aware_protos.aware.proto import messages_pb2

#----- Utilities -----

_CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')
_SPECIALS = {
    "latDeg": "lat_deg",
    "lonDeg": "lon_deg",
    "trackNumber": "track_number",
    "flightId": "flight_id",
    "latLon": "lat_lon",
    "positionUpdated": "position_updated",
    "measurementId": "measurement_id",
    "transferType": "transfer_type",
    "clearanceType": "clearance_type",
    "flightId1": "flight_id1",
    "flightId2": "flight_id2",
    "modeUpdated": "mode_updated",
    "lengthSeconds": "length_seconds",
    "actionName": "action_name",
    "markType": "mark_type",
    "markVariant": "mark_variant",
    "markScope": "mark_scope",
    "markSet": "mark_set",
}

def _to_snake(s):
    if not isinstance(s, str):
        return s
    return _SPECIALS.get(s, _CAMEL_RE.sub('_', s).lower())

def _normalize_keys(obj):
    if isinstance(obj, Dict):
        return {_to_snake(k): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, List):
        return [_normalize_keys(x) for x in obj]
    return obj

def _get(d, *names):
    if not isinstance(d, Dict):
        return None
    for k in names:
        if k in d:
            return d[k]
    return None

# ----- Distance measurement helper -----

def _flatten_mp(prefix, mp, *, log_conflicts=False):
    out = {}
    if not isinstance(mp, Dict):
        return out

    fi = mp.get("flight_id") or {}

    tn_top = mp.get("track_number")
    tn_fi  = fi.get("track_number")
    tn = tn_top if tn_top is not None else tn_fi
    out[f"{prefix}_track_number"] = tn

    if log_conflicts and (tn_top is not None and tn_fi is not None and tn_top != tn_fi):
        out[f"{prefix}_track_number_conflict"] = True

    if isinstance(mp.get("lat_lon"), Dict):
        out[f"{prefix}_kind"] = "lat_lon"
        out[f"{prefix}_lat_deg"] = mp["lat_lon"].get("lat_deg")
        out[f"{prefix}_lon_deg"] = mp["lat_lon"].get("lon_deg")
    elif isinstance(fi, Dict) and fi:
        out[f"{prefix}_kind"] = "flight_id"
        for k, v in fi.items():
            out[f"{prefix}_flight_{k}"] = v
    else:
        out[f"{prefix}_kind"] = "unknown"

    return out

# ----- Route helpers -----
_COORD_RE = re.compile(r"^(\d{2})(\d{2})([NS])(\d{3})(\d{2})([EW])$")
def _norm_upper_underscore(s):
    if not isinstance(s, str):
        return None
    return s.strip().upper().replace(" ", "_").replace("-", "_")

def _parse_compact_coord(s):
    if not isinstance(s, str):
        return None, None
    m = _COORD_RE.match(s.strip())
    if not m:
        return None, None
    lat_deg = int(m.group(1))
    lat_min = int(m.group(2))
    lat_hem = m.group(3)
    lon_deg = int(m.group(4))
    lon_min = int(m.group(5))
    lon_hem = m.group(6)

    lat = lat_deg + lat_min / 60.0
    lon = lon_deg + lon_min / 60.0
    if lat_hem == "S":
        lat = -lat
    if lon_hem == "W":
        lon = -lon
    return lat, lon

# ----- ASD ROWS EXTRACTOR -----

def rows_mouse_position(asd_dict):
    prefix = "mouse_position"
    if "mouse_position" not in asd_dict:
        return None
    p = asd_dict["mouse_position"]
    return {
        "event_name": "mouse_position",
        f"{prefix}_x": p.get("x"),
        f"{prefix}_y": p.get("y"),
    }

def rows_track_screen_position(asd_dict):
    prefix = "track_screen_position"
    if "track_screen_position" not in asd_dict:
        return None
    p = asd_dict["track_screen_position"]
    return {
        "event_name": "track_screen_position",
        f"{prefix}_x": p.get("x"),
        f"{prefix}_y": p.get("y"),
        f"{prefix}_track_number": p.get("track_number"),
        f"{prefix}_visible": p.get("visible"),
        **{f"{prefix}_flight_{k}": v for k, v in (p.get("flight_id") or {}).items()},
    }

def rows_track_label_position(asd_dict):
    prefix = "track_label_position"
    if "track_label_position" not in asd_dict:
        return None
    p = asd_dict["track_label_position"]
    out = {
        "event_name": "track_label_position",
        f"{prefix}_x": p.get("x"),
        f"{prefix}_y": p.get("y"),
        f"{prefix}_width": p.get("width"),
        f"{prefix}_height": p.get("height"),
        f"{prefix}_visible": p.get("visible"),
        f"{prefix}_hovered": p.get("hovered"),
        f"{prefix}_selected": p.get("selected"),
        f"{prefix}_on_pip": p.get("on_pip"),
        f"{prefix}_track_number": p.get("track_number"),
        **{f"{prefix}_flight_{k}": v for k, v in (p.get("flight_id") or {}).items()},
    }
    return out

def rows_speed_vector(asd_dict):
    prefix = "speed_vector"
    sv = asd_dict.get("speed_vector")
    if not isinstance(sv, Dict):
        return None

    if "mode_updated" in sv:
        return {
            "event_name": "speed_vector",
            f"{prefix}_variant": "mode_updated",
            f"{prefix}_mode_name": sv["mode_updated"].get("mode"),
        }
    if "visibility" in sv:
        v = sv["visibility"]
        tn = v.get("track_number") or (v.get("flight_id") or {}).get("track_number")
        vis = v.get("visible")
        return {
            "event_name": "speed_vector",
            f"{prefix}_variant": "visibility",
            f"{prefix}_track_number": tn,
            f"{prefix}_visible": vis,
            f"{prefix}_visibility_event_type": (
                "set_true" if vis is True else "set_false" if vis is False else "touched"
            ),
            **{f"{prefix}_flight_{k}": v for k, v in (v.get("flight_id") or {}).items()},
        }
    if "length" in sv:
        return {
            "event_name": "speed_vector",
            f"{prefix}_variant": "length",
            f"{prefix}_length_seconds": sv["length"].get("length_seconds"),
        }
    return None

def rows_popup(asd_dict):
    prefix = "popup"
    p = asd_dict.get("popup")
    if not isinstance(p, Dict):
        return None
    tn = p.get("track_number") or (p.get("flight_id") or {}).get("track_number")
    base = {
        "event_name": "popup",
        f"{prefix}_name": p.get("name"),
        f"{prefix}_opened": p.get("opened"),
        f"{prefix}_track_number": tn,
    }
    base.update({f"{prefix}_flight_{k}": v for k, v in (p.get("flight_id") or {}).items()})
    return base

def rows_transfer(asd_dict):
    prefix = "transfer"
    t = asd_dict.get("transfer")
    if not isinstance(t, Dict):
        return None
    tn = t.get("track_number") or (t.get("flight_id") or {}).get("track_number")
    base = {
        "event_name": "transfer",
        "transfer_type_name": t.get("transfer_type"),
        f"{prefix}_track_number": tn,
    }
    base.update({f"{prefix}_flight_{k}": v for k, v in (t.get("flight_id") or {}).items()})
    return base

def rows_clearance(asd_dict):
    prefix = "clearance"
    c = asd_dict.get("clearance")
    if not isinstance(c, Dict):
        return None
    tn = c.get("track_number") or (c.get("flight_id") or {}).get("track_number")
    base = {
        "event_name": "clearance",
        "clearance_type": c.get("clearance_type"),
        "clearance": c.get("clearance"),
        f"{prefix}_track_number": tn,
    }
    base.update({f"{prefix}_flight_{k}": v for k, v in (c.get("flight_id") or {}).items()})
    return base

def rows_distance_measurement(asd_dict):
    prefix = "distance_measurement"
    dm = _get(asd_dict, "distance_measurement", "distanceMeasurement")
    if not isinstance(dm, Dict):
        return None

    row = {"event_name": "distance_measurement"}

    added = _get(dm, "added", "added")
    if isinstance(added, Dict):
        row[f"{prefix}_change"] = "added"
        row[f"{prefix}_measurement_id"] = _get(added, "measurement_id", "measurementId")
        first = _get(added, "first", "first")
        second = _get(added, "second", "second")
        if first:
            row.update(_flatten_mp(f"{prefix}_first", first))
        if second:
            row.update(_flatten_mp(f"{prefix}_second", second))
        return row

    pos = _get(dm, "position_updated", "positionUpdated")
    if isinstance(pos, Dict):
        row[f"{prefix}_change"] = "position_updated"
        row[f"{prefix}_measurement_id"] = _get(pos, "measurement_id", "measurementId")
        start = _get(pos, "start", "start") or {}
        end = _get(pos, "end", "end") or {}
        row[f"{prefix}_start_x"] = _get(start, "x", "x")
        row[f"{prefix}_start_y"] = _get(start, "y", "y")
        row[f"{prefix}_end_x"] = _get(end, "x", "x")
        row[f"{prefix}_end_y"] = _get(end, "y", "y")
        return row

    removed = _get(dm, "removed", "removed")
    if isinstance(removed, Dict):
        row[f"{prefix}_change"] = "removed"
        row[f"{prefix}_measurement_id"] = _get(removed, "measurement_id", "measurementId")
        return row

    return None

def rows_sep_tool(asd_dict):
    prefix = "sep_tool"
    st = _get(asd_dict, "sep_tool", "sepTool")
    if not isinstance(st, Dict):
        return None

    row = {"event_name": "sep_tool"}
    row[f"{prefix}_type"] = _get(st, "type", "type")

    opened = _get(st, "opened", "opened")
    if isinstance(opened, Dict):
        row[f"{prefix}_change"] = "opened"
        fi = _get(opened, "flight_id", "flightId") or {}
        row[f"{prefix}_opened_track_number"] = _get(fi, "track_number", "trackNumber")
        if isinstance(fi, Dict):
            for k, v in fi.items():
                row[f"{prefix}_opened_flight_{_to_snake(k)}"] = v
        return row

    connected = _get(st, "connected", "connected")
    if isinstance(connected, Dict):
        row[f"{prefix}_change"] = "connected"
        fi1 = _get(connected, "flight_id1", "flightId1") or {}
        fi2 = _get(connected, "flight_id2", "flightId2") or {}
        row[f"{prefix}_connected_track_number_1"] = _get(fi1, "track_number", "trackNumber")
        row[f"{prefix}_connected_track_number_2"] = _get(fi2, "track_number", "trackNumber")
        if isinstance(fi1, Dict):
            for k, v in fi1.items():
                row[f"{prefix}_connected_flight1_{_to_snake(k)}"] = v
        if isinstance(fi2, Dict):
            for k, v in fi2.items():
                row[f"{prefix}_connected_flight2_{_to_snake(k)}"] = v
        return row

    if "closed" in st:
        row[f"{prefix}_change"] = "closed"
        row[f"{prefix}_closed"] = bool(_get(st, "closed", "closed"))
        return row

    row[f"{prefix}_change"] = None
    return row

def rows_route_interaction(asd_dict):
    prefix = "route_interaction"
    ri = _get(asd_dict, "route_interaction", "routeInteraction")
    if not isinstance(ri, dict):
        return None

    row = {"event_name": "route_interaction"}
    action_raw = _get(ri, "action_type", "actionType")
    row[f"{prefix}_action_type_raw"] = action_raw
    row[f"{prefix}_action_type_name"] = _norm_upper_underscore(action_raw)

    val = _get(ri, "value", "value")
    row[f"{prefix}_value"] = val
    lat, lon = _parse_compact_coord(val)
    row[f"{prefix}_value_lat_deg"] = lat
    row[f"{prefix}_value_lon_deg"] = lon
    row[f"{prefix}_value_kind"] = (
        "coord" if lat is not None else ("fix" if isinstance(val, str) else None)
    )

    fi = _get(ri, "flight_id", "flightId") or {}
    tn = _get(ri, "track_number", "trackNumber") or _get(fi, "track_number", "trackNumber")
    row[f"{prefix}_track_number"] = tn

    if isinstance(fi, dict):
        for k, v in fi.items():
            row[f"{prefix}_flight_{_to_snake(k)}"] = v

    return row

def rows_keyboard_shortcut(asd_dict):
    prefix = "keyboard"
    ks = _get(asd_dict, "keyboard_shortcut", "keyboardShortcut")
    if not isinstance(ks, dict):
        return None
    name = _get(ks, "action_name", "actionName")
    return {
        "event_name": "keyboard_shortcut",
        f"{prefix}_action_name": name,
        f"{prefix}_action_name_norm": _norm_upper_underscore(name),
    }

def rows_mark(asd_dict):
    prefix = "track_mark"
    m = _get(asd_dict, "track_mark", "trackMark", "mark")
    if not isinstance(m, dict):
        return None

    out = {"event_name": "track_mark"}
    out["mark_type_raw"] = _get(m, "mark_type", "markType")
    out["mark_variant_raw"] = _get(m, "mark_variant", "markVariant")
    out["mark_scope_raw"] = _get(m, "mark_scope", "markScope")
    out["mark_set"] = _get(m, "mark_set", "markSet")

    out["mark_type_name"] = _norm_upper_underscore(out["mark_type_raw"])
    out["mark_variant_name"] = _norm_upper_underscore(out["mark_variant_raw"])
    out["mark_scope_name"] = _norm_upper_underscore(out["mark_scope_raw"])

    if out["mark_set"] is True:
        out["mark_action"] = "SET"
    elif out["mark_set"] is False:
        out["mark_action"] = "UNSET"
    else:
        out["mark_action"] = "TOUCH"

    tn = _get(m, "track_number", "trackNumber")
    fi = _get(m, "flight_id", "flightId") or {}
    if tn is None and isinstance(fi, dict):
        tn = _get(fi, "track_number", "trackNumber")
    out[f"{prefix}_track_number"] = tn

    if isinstance(fi, dict):
        for k, v in fi.items():
            out[f"{prefix}_flight_{_to_snake(k)}"] = v

    return out

EXTRACTORS = [
    rows_mouse_position,
    rows_track_screen_position,
    rows_track_label_position,
    rows_speed_vector,
    rows_popup,
    rows_transfer,
    rows_clearance,
    rows_distance_measurement,
    rows_sep_tool,
    rows_route_interaction,
    rows_keyboard_shortcut,
    rows_mark,
]

# ----- HIGH-LEVEL BUILDER -----
def iter_asd_events_from_db(db_path: Path, start_ms: int, end_ms: int, batch: int = 50_000) -> Iterable[Tuple[int, Dict]]:
    """
    Iterate (epoch_ms, normalized_asd_dict) from an open sqlite3 connection.
    Expects an 'events' table with protobuf messages of type Event in 'payload'.
    """
    
    uri = f"file:{db_path}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.text_factory = bytes
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA mmap_size=268435456")
    con.execute("PRAGMA temp_store=MEMORY")
    
    sql = (
        'SELECT epoch_ms, payload FROM "events" '
        'WHERE epoch_ms BETWEEN ? AND ? '
        'ORDER BY epoch_ms'
    )
    cur = con.execute(sql, (start_ms, end_ms))
    try:
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for ms, blob in rows:
                ev = messages_pb2.Event()
                ev.ParseFromString(blob)
                if ev.WhichOneof("payload") != "asd_event":
                    continue
                raw = MessageToDict(ev.asd_event, always_print_fields_with_no_presence=True, preserving_proto_field_name=True)
                yield int(ms), _normalize_keys(raw)
    finally:
        con.close()


def build_asd_frame_from_db(db_path: Path, start_ms: int, end_ms: int) -> "pd.DataFrame":
    """
    High-level: read ASD events from DB, apply all EXTRACTORS, return a flat DataFrame.
    (Import pandas in the caller or here, as you prefer.)
    """
    import pandas as pd 

    rows = []
    for ms, asd in iter_asd_events_from_db(db_path, start_ms, end_ms):
        for f in EXTRACTORS:
            r = f(asd)
            if r is not None:
                r["epoch_ms"] = ms
                rows.append(r)
                break  # only one rows_* should match each asd_event

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("epoch_ms").reset_index(drop=True)