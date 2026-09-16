"""Atomic write-once journal; no mutable latest pointer is trusted on resume."""
from pathlib import Path
from datetime import datetime, timezone

from .artifacts import publish_json_once, read_verified_json
from .contracts import ContractViolation
from .semantic_experiment_freeze import _hash


class PilotJournal:
    def __init__(self, directory, binding):
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ContractViolation("JOURNAL_SYMLINK")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binding = dict(binding)
        self.events = []
        tip = "0" * 64
        for index, path in enumerate(sorted(self.directory.glob("*.json"))):
            if path.name != f"{index:06d}.json" or path.is_symlink():
                raise ContractViolation("JOURNAL_SEQUENCE_BROKEN")
            row = read_verified_json(path)
            if (row.get("index") != index or row.get("previous_sha256") != tip
                or row.get("binding") != self.binding
                or row.get("event_sha256") != _hash({k:v for k,v in row.items() if k != "event_sha256"})):
                raise ContractViolation("JOURNAL_CHAIN_OR_BINDING_MISMATCH")
            self.events.append(row)
            tip = row["event_sha256"]

    def append(self, kind, data):
        core = {"schema_version": "pilot-journal-v1", "index": len(self.events),
            "previous_sha256": self.events[-1]["event_sha256"] if self.events else "0"*64,
            "binding": self.binding, "kind": kind, "data": data,
            "utc": datetime.now(timezone.utc).isoformat()}
        row = {**core, "event_sha256": _hash(core)}
        publish_json_once(self.directory / f"{len(self.events):06d}.json", row)
        self.events.append(row)
        return row

    def latest(self, kind, root=None, phase=None):
        return next((e["data"] for e in reversed(self.events)
            if e["kind"] == kind and (root is None or e["data"].get("root_id") == root)
            and (phase is None or e["data"].get("phase") == phase)), None)
