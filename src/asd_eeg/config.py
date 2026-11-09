"""Validated, portable configuration shared by training and inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Node:
    name: str
    channels: tuple[str, ...]


@dataclass(frozen=True)
class Band:
    name: str
    low: float
    high: float


@dataclass(frozen=True)
class Preprocessing:
    target_sfreq: float = 128.0
    low_hz: float = 1.0
    high_hz: float = 40.0
    line_hz: float = 50.0
    reference: str = "average"
    window_seconds: float = 8.0
    min_windows: int = 3
    max_windows: int = 20
    max_peak_to_peak_uv: float = 300.0


@dataclass(frozen=True)
class Circuit:
    passes: int = 2
    bandwidth: float = 1.0
    edge_seed: int = 1729


@dataclass(frozen=True)
class Evaluation:
    folds: int = 5
    seed: int = 42
    svm_c: float = 1.0


@dataclass(frozen=True)
class Config:
    nodes: tuple[Node, ...]
    bands: tuple[Band, ...]
    preprocessing: Preprocessing
    circuit: Circuit
    evaluation: Evaluation

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((i, j) for i in range(len(self.nodes)) for j in range(i + 1, len(self.nodes)))

    @property
    def n_features(self) -> int:
        return 4 * len(self.nodes) + 2 * len(self.edges)

    @property
    def id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        config = cls(
            nodes=tuple(Node(n["name"], tuple(n["channels"])) for n in raw["nodes"]),
            bands=tuple(Band(**b) for b in raw["bands"]),
            preprocessing=Preprocessing(**raw.get("preprocessing", {})),
            circuit=Circuit(**raw.get("circuit", {})),
            evaluation=Evaluation(**raw.get("evaluation", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        p = self.preprocessing
        if not 2 <= len(self.nodes) <= 6:
            raise ValueError("Configure between two and six verified EEG nodes.")
        names = [n.name for n in self.nodes]
        channels = [c.casefold() for n in self.nodes for c in n.channels]
        if len(set(names)) != len(names) or any(not n.channels or not n.name for n in self.nodes):
            raise ValueError("Each node needs a unique name and at least one channel.")
        if len(channels) != len(set(channels)):
            raise ValueError("A channel cannot belong to more than one node.")
        if tuple(b.name for b in self.bands) != ("delta", "theta", "alpha", "beta"):
            raise ValueError("Bands must be ordered delta, theta, alpha, beta.")
        numbers = [p.target_sfreq, p.low_hz, p.high_hz, p.line_hz, p.window_seconds,
                   p.max_peak_to_peak_uv, self.circuit.bandwidth, self.evaluation.svm_c]
        if not np.isfinite(numbers).all():
            raise ValueError("Configuration numbers must be finite.")
        if not 0 < p.low_hz < p.high_hz < p.target_sfreq / 2:
            raise ValueError("Filter limits must lie below the target Nyquist frequency.")
        for band in self.bands:
            if not np.isfinite([band.low, band.high]).all() or not (
                p.low_hz <= band.low < band.high <= p.high_hz
            ):
                raise ValueError("Every band must lie within the preprocessing passband.")
        if p.reference not in {"average", "none"}:
            raise ValueError("Reference must be 'average' or 'none'.")
        if p.window_seconds <= 0 or p.min_windows < 1 or p.max_windows < p.min_windows:
            raise ValueError("Check the window duration and minimum/maximum window counts.")
        if p.max_peak_to_peak_uv <= 0 or p.line_hz < 0:
            raise ValueError("QC amplitude must be positive; notch frequency cannot be negative.")
        if self.circuit.passes not in range(1, 5) or self.circuit.bandwidth <= 0:
            raise ValueError("Use one to four encoding passes and a positive bandwidth.")
        if self.evaluation.folds < 2 or self.evaluation.svm_c <= 0:
            raise ValueError("Evaluation requires at least two folds and a positive SVM C.")


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    return Config.from_dict(raw)
