"""Unit tests for scripts/run_benchmarks.py."""

from __future__ import annotations

import json
from pathlib import Path
from scripts.run_benchmarks import main


def test_run_benchmarks_subset(tmp_path: Path) -> None:
    out_dir = tmp_path / "bench"
    rc = main(["--out", str(out_dir), "--experiments", "A1", "A3"])
    assert rc == 0
    data = json.loads((out_dir / "ablation_results.json").read_text(encoding="utf-8"))
    assert "A1" in data and "A3" in data
