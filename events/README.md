# Read Polaris events

This is an example script that shows how to decode the binary data the proxy is writing to disk.
You will very likely want to modify it to filter out information you don't care about; for example to only write flight records for a single flight, or only write Cat062 messages, or so on.

## Setup

Run

```
./setup.sh
```

to create a Python virtual environment, activate it, install the `protobuf` and `asterix_decode` packages in the environment, and generate protobuf classes.
This only has to happen once.

To reuse an environment you've already created, run `source env/bin/activate`.

## Run

Activate the virtual environment if it isn't active already:

```
source env/bin/activate
```

To print help:

```
./read-events.py --help
```

To dump all the events in a database file:

```
./read-events FILENAME
```

The file should be able to read the old database files (where the payloads are in JSON format) as well by adding the `--json` flag.

## Tern protobuf files

The various Tern protobuf files contain some documentation about what the various fields mean.
That may or may not be helpful.
Don't hesitate to ask when something is not clear.
