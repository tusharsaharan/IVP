import json
import os
import tempfile
from pathlib import Path


def prediction_json(prediction: dict) -> str:
    """The same strict, UTF-8 JSON representation for CLI and app downloads."""
    return json.dumps(prediction, indent=2, ensure_ascii=False, allow_nan=False)


def export_prediction(prediction: dict, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("Prediction output must use the .json extension.")
    payload = prediction_json(prediction)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize first, then atomically replace: failed exports leave earlier work intact.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
