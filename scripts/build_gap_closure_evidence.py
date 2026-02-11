#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from research.gap_closure import run_full_gap_closure


def main() -> int:
    summary = run_full_gap_closure()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
