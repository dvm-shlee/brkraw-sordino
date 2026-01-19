#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path("release_metadata.json")
    if not path.exists():
        raise SystemExit("release_metadata.json not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    tag = data.get("tag", "")
    if not tag:
        raise SystemExit("Tag not found in release metadata")
    print(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
