from pathlib import Path

import joblib
import numpy as np

from asd_eeg.config import Config
from asd_eeg.training.model import MODELS, FittedModel


def _subject_ids(value, name: str) -> list[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(sid, str) or not sid.strip() for sid in value)):
        raise ValueError(f"Bundle {name} must contain nonempty subject IDs.")
    if len(set(value)) != len(value):
        raise ValueError(f"Bundle has duplicate {name}.")
    return value


def _restore_bundle(path: str | Path) -> dict:
    """Restore a trusted local project bundle, without reordering references.

    Validation is not a serialization sandbox. Never load an uploaded or untrusted
    joblib file: deserialization can execute Python code before these checks.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ValueError("Could not restore the trusted model bundle. Re-export it with this package.") from exc
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported model-bundle schema. Re-export with this package.")
    model = bundle.get("model")
    if not isinstance(model, FittedModel):
        raise ValueError("The bundle does not contain a compatible fitted model.")
    meta = bundle.get("metadata")
    required = {"model", "fold", "run_id", "dataset", "data_kind", "config_id",
                "feature_fingerprint", "train_subject_ids", "test_subject_ids", "class_mapping"}
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError("Model-bundle metadata is incomplete. Re-export with this package.")
    train = _subject_ids(meta["train_subject_ids"], "training references")
    test = _subject_ids(meta["test_subject_ids"], "held-out subjects")
    if set(train) & set(test):
        raise ValueError("Bundle has overlapping training/test subjects.")
    if meta["model"] not in MODELS or model.name != meta["model"]:
        raise ValueError("Bundle model name is inconsistent with its fitted model.")
    if (not isinstance(meta["fold"], int) or isinstance(meta["fold"], bool) or meta["fold"] < 0
            or meta["data_kind"] not in {"recorded", "synthetic"}
            or any(not isinstance(meta[key], str) or not meta[key].strip()
                   for key in ("dataset", "run_id", "config_id", "feature_fingerprint"))):
        raise ValueError("Bundle provenance metadata is invalid.")
    if not isinstance(model.config, Config):
        raise ValueError("Bundle pipeline configuration is incompatible.")
    model.config.validate()
    angles = np.asarray(model.train_angles)
    width = model.config.n_features
    if (angles.shape != (len(train), width) or not np.isfinite(angles).all()
            or meta["config_id"] != model.config.id):
        raise ValueError("Bundle reference order or configuration is inconsistent.")
    reference = np.asarray(model.feature_reference)
    if reference.shape != (width,) or not np.isfinite(reference).all():
        raise ValueError("Bundle training-derived feature reference is invalid.")
    if model.name in {"quantum", "reassigned"}:
        states = np.asarray(model.train_states)
        if states.shape != (len(train), 2 ** len(model.config.nodes)) or not np.isfinite(states).all():
            raise ValueError("Reference states do not align with training subjects.")
        if not np.allclose(np.sum(np.abs(states) ** 2, axis=1), 1.0, atol=1e-8):
            raise ValueError("Bundle reference states are not normalized.")
    elif model.train_states is not None:
        raise ValueError("Non-quantum bundle contains unexpected reference states.")
    for name in ("mean_", "scale_", "var_"):
        values = np.asarray(getattr(model.scaler, name, None))
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError("Bundle training-fitted scaler is incomplete or invalid.")
    if np.any(model.scaler.scale_ <= 0) or np.any(model.scaler.var_ < 0):
        raise ValueError("Bundle training-fitted scaler has invalid scales.")
    expected_width = width if model.name == "rbf" else len(train)
    if getattr(model.estimator, "n_features_in_", None) != expected_width:
        raise ValueError("Bundle SVM input width does not match the full training-reference order.")
    expected_kernel = "rbf" if model.name == "rbf" else "precomputed"
    if getattr(model.estimator, "kernel", None) != expected_kernel:
        raise ValueError("Bundle SVM kernel does not match the saved model.")
    if not np.array_equal(getattr(model.estimator, "classes_", None), [0, 1]):
        raise ValueError("Bundle SVM class order does not support the declared score direction.")
    if meta["class_mapping"] != {0: "Control", 1: "ASD"}:
        raise ValueError("Unsupported class mapping.")
    return bundle


def load_bundle(path: str | Path) -> dict:
    """Load only trusted project-generated files, returning actionable restore errors."""
    try:
        return _restore_bundle(path)
    except (TypeError, AttributeError, KeyError, IndexError) as exc:
        raise ValueError("Model bundle contains malformed or incompatible fitted state. Re-export it.") from exc


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
