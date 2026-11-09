from pathlib import Path

import joblib

from asd_eeg.training.model import FittedModel


def load_bundle(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported model-bundle schema. Re-export with this package.")
    model = bundle.get("model")
    if not isinstance(model, FittedModel):
        raise ValueError("The bundle does not contain a compatible fitted model.")
    meta = bundle["metadata"]
    train, test = meta["train_subject_ids"], meta["test_subject_ids"]
    if len(set(train)) != len(train) or set(train) & set(test):
        raise ValueError("Bundle has duplicate references or overlapping training/test subjects.")
    if len(train) != len(model.train_angles) or meta["config_id"] != model.config.id:
        raise ValueError("Bundle reference order or configuration is inconsistent.")
    if model.train_states is not None and len(model.train_states) != len(train):
        raise ValueError("Reference states do not align with training subjects.")
    if meta["class_mapping"] != {0: "Control", 1: "ASD"}:
        raise ValueError("Unsupported class mapping.")
    return bundle


def check_subject(bundle: dict, subject_id: str, dataset: str, require_held_out: bool = True) -> str:
    meta = bundle["metadata"]
    if dataset != meta["dataset"]:
        raise ValueError("Recording dataset differs from the model dataset. Cross-dataset inference needs review.")
    if subject_id in meta["train_subject_ids"]:
        raise ValueError("This subject was used for training. Select its held-out fold bundle.")
    if subject_id in meta["test_subject_ids"]:
        return "held-out"
    if require_held_out:
        raise ValueError("This subject is not in the bundle's held-out fold. Select the matching bundle.")
    return "new-subject"
