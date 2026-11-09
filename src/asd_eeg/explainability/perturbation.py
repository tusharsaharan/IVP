import numpy as np


def connection_sensitivity(bundle: dict, feature_vector: np.ndarray) -> list[dict]:
    model = bundle["model"]
    config = model.config
    original = float(model.predict(feature_vector[None, :])[1][0])
    rows = []
    for index, (i, j) in enumerate(config.edges):
        changed = feature_vector.copy()
        positions = [4 * len(config.nodes) + index, 4 * len(config.nodes) + len(config.edges) + index]
        changed[positions] = model.feature_reference[positions]
        score = float(model.predict(changed[None, :])[1][0])
        rows.append({"connection": f"{config.nodes[i].name} / {config.nodes[j].name}",
                     "original_score": original, "perturbed_score": score,
                     "score_change": score - original})
    return rows
