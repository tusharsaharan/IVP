import json
from pathlib import Path


def export_prediction(prediction: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prediction, indent=2, allow_nan=False), encoding="utf-8")
    return path
