"""JSON 出力。"""

from __future__ import annotations

import json
from pathlib import Path

from chordscope.models import TrackAnalysis


def write_json(track: TrackAnalysis, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(track.model_dump_json())
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
