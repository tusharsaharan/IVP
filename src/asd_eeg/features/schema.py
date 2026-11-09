"""The fixed, label-free feature contract for subject-level EEG graphs."""

from __future__ import annotations

from asd_eeg.config import Config

FEATURE_SCHEMA_VERSION = 1

POWER_DEFINITION = (
    "Median across used EEG windows of log10 relative band power. Relative power for an "
    "electrode is the integrated Welch PSD inside the band divided by the integrated 1-30 Hz "
    "PSD. Electrode values are averaged within the node before the log transform."
)
CONNECTIVITY_DEFINITION = (
    "Pearson correlation between two regional log alpha-envelope power series inside one "
    "accepted window. A regional envelope averages the electrode envelopes assigned to the node."
)
WINDOW_AGGREGATION = (
    "Accepted windows are pooled across the subject's recordings in recording_id order, then "
    "evenly subsampled to the configured maximum. Powers use the median across used windows; "
    "edge correlations use the median and the 75th-minus-25th percentile IQR."
)


def feature_names(config: Config) -> tuple[str, ...]:
    """Return the authoritative feature order for a configuration."""
    names = [f"{node.name}: {band.name} log-relative power"
             for node in config.nodes for band in config.bands]
    for statistic in ("median", "IQR"):
        names.extend(f"{config.nodes[i].name} / {config.nodes[j].name}: connectivity {statistic}"
                     for i, j in config.edges)
    return tuple(names)


def feature_schema(config: Config) -> dict:
    """Describe every feature, its source, units, and aggregation rule."""
    names = feature_names(config)
    features: list[dict] = []
    for node_index, node in enumerate(config.nodes):
        for band in config.bands:
            index = len(features)
            features.append({
                "index": index,
                "name": names[index],
                "group": "band_power",
                "kind": "log10_relative_band_power",
                "node_index": node_index,
                "node": node.name,
                "source_channels": list(node.channels),
                "band": band.name,
                "band_low_hz": band.low,
                "band_high_hz": band.high,
                "statistic": "median",
                "units": "log10(ratio)",
                "definition": POWER_DEFINITION,
            })
    for statistic in ("median", "IQR"):
        for edge_index, (left, right) in enumerate(config.edges):
            index = len(features)
            features.append({
                "index": index,
                "name": names[index],
                "group": "connectivity_median" if statistic == "median" else "connectivity_iqr",
                "kind": "alpha_envelope_pearson_r",
                "edge_index": edge_index,
                "left_node_index": left,
                "right_node_index": right,
                "left_node": config.nodes[left].name,
                "right_node": config.nodes[right].name,
                "left_channels": list(config.nodes[left].channels),
                "right_channels": list(config.nodes[right].channels),
                "statistic": statistic,
                "units": "unitless Pearson r",
                "definition": CONNECTIVITY_DEFINITION,
            })
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "config_id": config.id,
        "feature_count": config.n_features,
        "label_free": True,
        "node_order": [{"index": i, "name": node.name, "channels": list(node.channels)}
                       for i, node in enumerate(config.nodes)],
        "edge_order": [{"index": k, "left_node_index": i, "right_node_index": j,
                        "left_node": config.nodes[i].name, "right_node": config.nodes[j].name}
                       for k, (i, j) in enumerate(config.edges)],
        "bands": [{"name": band.name, "low_hz": band.low, "high_hz": band.high}
                  for band in config.bands],
        "power_definition": POWER_DEFINITION,
        "connectivity_definition": CONNECTIVITY_DEFINITION,
        "window_aggregation": WINDOW_AGGREGATION,
        "features": features,
    }
