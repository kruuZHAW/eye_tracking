#!/usr/bin/env python3

# Reads a Polaris event file and prints the events.

# Needs the google.protobuf Python package to work.
# Install it in a virtual environment with:
#
# $ python3 -m venv env
# $ source env/bin/activate
# $ pip install -r requirements.txt

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import asterix
import google.protobuf.json_format

from aware_protos.aware.proto import messages_pb2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("events", help="Sqlite3 file with Polaris events")
    parser.add_argument("--json", action="store_true", help="Assume events are encoded in JSON format")
    args = parser.parse_args()

    with closing(sqlite3.connect(f"{Path(args.events).as_uri()}?immutable=1", uri=True)) as db:
        if not args.json:
            db.text_factory = bytes

        for row in db.execute("SELECT epoch_ms, payload FROM events"):
            if args.json:
                event = google.protobuf.json_format.Parse(row[1], messages_pb2.Event())
            else:
                event = messages_pb2.Event()
                event.ParseFromString(row[1])

            if event.HasField("asterix"):
                ast = asterix.parse(event.asterix)
                print(ast)
            else:
                print(event)
