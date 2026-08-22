# -*- coding: utf-8 -*-
"""Pure clustering helpers and a short-lived server-side result cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


MAX_PROJECTION_POINTS = 6000
MAX_DIAGNOSTIC_ROWS = 12000
MAX_SILHOUETTE_ROWS = 5000


@dataclass
class CachedClusteringResult:
    frame: pd.DataFrame
    analysis: dict
    committed_step: dict
    signature: str


_RESULT_CACHE: dict[str, CachedClusteringResult] = {}


def clustering_signature(**parameters) -> str:
    """Return a stable signature for inputs that materially affect a run."""
    normalized = json.dumps(parameters, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_result(result: CachedClusteringResult) -> str:
    reference = uuid4().hex
    _RESULT_CACHE[reference] = result
    return reference


def cached_result(reference) -> CachedClusteringResult | None:
    return _RESULT_CACHE.get(str(reference or ""))


def discard_cached_result(reference) -> None:
    _RESULT_CACHE.pop(str(reference or ""), None)


def _unique_name(columns, base):
    name = str(base or "Кластер").strip() or "Кластер"
    if name not in columns:
        return name
    index = 2
    while f"{name}_{index}" in columns:
        index += 1
    return f"{name}_{index}"


def _scaler(name):
    return {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "none": None,
    }.get(name or "standard", StandardScaler())


def _prepare_matrix(frame, features, missing_policy, scaling):
    selected = list(dict.fromkeys(str(column) for column in (features or [])))
    selected = [column for column in selected if column in frame.columns]
    if len(selected) < 2:
        raise ValueError("Выберите не меньше двух числовых каналов.")

    numeric = frame[selected].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    all_missing = [column for column in selected if numeric[column].notna().sum() == 0]
    if all_missing:
        joined = ", ".join(all_missing[:4])
        raise ValueError(f"Нет числовых значений в каналах: {joined}.")

    if missing_policy == "drop":
        valid_mask = numeric.notna().all(axis=1)
        prepared = numeric.loc[valid_mask].copy()
    else:
        valid_mask = pd.Series(True, index=frame.index)
        prepared = numeric.copy()
        if missing_policy == "zero":
            prepared = prepared.fillna(0)
        else:
            reducer = prepared.mean if missing_policy == "mean" else prepared.median
            prepared = prepared.fillna(reducer())

    if len(prepared) < 3:
        raise ValueError("После обработки пропусков осталось меньше трёх строк.")

    constant = [column for column in selected if prepared[column].nunique(dropna=True) <= 1]
    used = [column for column in selected if column not in constant]
    if len(used) < 2:
        raise ValueError("Для кластеризации нужны минимум два изменяющихся числовых канала.")
    prepared = prepared[used]

    transformer = _scaler(scaling)
    matrix = prepared.to_numpy(dtype=float, copy=True)
    if transformer is not None:
        matrix = transformer.fit_transform(matrix)
    return prepared.index, used, constant, matrix


def _model(algorithm, k, row_count, random_state):
    if algorithm == "minibatch":
        return MiniBatchKMeans(
            n_clusters=k,
            n_init=10,
            random_state=random_state,
            batch_size=min(4096, max(256, row_count)),
        )
    return KMeans(n_clusters=k, n_init=10, random_state=random_state)


def _sample_indices(row_count, limit, random_state):
    if row_count <= limit:
        return np.arange(row_count)
    generator = np.random.default_rng(random_state)
    return np.sort(generator.choice(row_count, size=limit, replace=False))


def _quality_metrics(matrix, labels, inertia, random_state):
    unique = np.unique(labels)
    metrics = {
        "inertia": float(inertia),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if len(unique) < 2 or len(unique) >= len(matrix):
        return metrics
    try:
        silhouette_rows = min(MAX_SILHOUETTE_ROWS, len(matrix))
        metrics["silhouette"] = float(
            silhouette_score(
                matrix,
                labels,
                sample_size=silhouette_rows if silhouette_rows < len(matrix) else None,
                random_state=random_state,
            )
        )
    except ValueError:
        pass
    metric_idx = _sample_indices(len(matrix), MAX_DIAGNOSTIC_ROWS, random_state)
    metric_matrix = matrix[metric_idx]
    metric_labels = labels[metric_idx]
    if 2 <= len(np.unique(metric_labels)) < len(metric_matrix):
        try:
            metrics["davies_bouldin"] = float(
                davies_bouldin_score(metric_matrix, metric_labels)
            )
        except ValueError:
            pass
        try:
            metrics["calinski_harabasz"] = float(
                calinski_harabasz_score(metric_matrix, metric_labels)
            )
        except ValueError:
            pass
    return metrics


def _diagnostics(matrix, algorithm, random_state):
    sample_idx = _sample_indices(len(matrix), MAX_DIAGNOSTIC_ROWS, random_state)
    sample = matrix[sample_idx]
    upper = min(12, len(sample) - 1)
    if upper < 2:
        return {"ks": [], "inertias": [], "silhouettes": [], "recommended_k": None}

    ks, inertias, silhouettes = [], [], []
    for candidate in range(2, upper + 1):
        ks.append(candidate)
        inertia_value = None
        silhouette_value = None
        try:
            fitted = _model(algorithm, candidate, len(sample), random_state).fit(sample)
            labels = fitted.labels_
            inertia_value = float(fitted.inertia_)
            if 2 <= len(np.unique(labels)) < len(sample):
                silhouette_rows = min(MAX_SILHOUETTE_ROWS, len(sample))
                silhouette_value = float(silhouette_score(
                    sample,
                    labels,
                    sample_size=silhouette_rows if silhouette_rows < len(sample) else None,
                    random_state=random_state,
                ))
        except (TypeError, ValueError):
            pass
        inertias.append(inertia_value)
        silhouettes.append(silhouette_value)
    valid_silhouettes = [
        (value, candidate)
        for candidate, value in zip(ks, silhouettes)
        if value is not None and np.isfinite(value)
    ]
    recommended = max(valid_silhouettes)[1] if valid_silhouettes else None
    return {
        "ks": ks,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "recommended_k": recommended,
    }


def run_clustering(
    frame,
    *,
    features,
    k=4,
    algorithm="kmeans",
    scaling="standard",
    missing_policy="drop",
    output_column="Кластер",
    include_id=True,
    include_distance=True,
    include_pca=True,
    random_state=42,
    signature="",
):
    """Fit a clustering model and return an aligned augmented dataframe."""
    if frame is None or frame.empty:
        raise ValueError("Dataset пуст.")
    try:
        k = int(k)
    except (TypeError, ValueError) as error:
        raise ValueError("K должно быть целым числом.") from error

    valid_index, used, constant, matrix = _prepare_matrix(
        frame, features, missing_policy, scaling
    )
    if k < 2:
        raise ValueError("K должно быть не меньше двух.")
    if k >= len(matrix):
        raise ValueError(f"K={k} должно быть меньше количества строк ({len(matrix)}).")

    fitted = _model(algorithm, k, len(matrix), int(random_state)).fit(matrix)
    labels = fitted.labels_.astype(int)
    actual_clusters = len(np.unique(labels))
    if actual_clusters < 2:
        raise ValueError("Модель нашла меньше двух различимых кластеров.")
    distances = fitted.transform(matrix).min(axis=1)
    projection_model = PCA(n_components=2, random_state=int(random_state))
    projection = projection_model.fit_transform(matrix)

    result = frame.copy()
    cluster_column = _unique_name(result.columns, output_column)
    result[cluster_column] = pd.Series(pd.NA, index=result.index, dtype="string")
    display_labels = pd.Series(
        [f"Кластер {label + 1}" for label in labels],
        index=valid_index,
        dtype="string",
    )
    result.loc[valid_index, cluster_column] = display_labels
    outputs = [cluster_column]

    id_column = distance_column = pca_x_column = pca_y_column = None
    if include_id:
        id_column = _unique_name(result.columns, f"{cluster_column}_ID")
        result[id_column] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result.loc[valid_index, id_column] = pd.Series(labels + 1, index=valid_index, dtype="Int64")
        outputs.append(id_column)
    if include_distance:
        distance_column = _unique_name(result.columns, f"{cluster_column}_Расстояние")
        result[distance_column] = np.nan
        result.loc[valid_index, distance_column] = distances
        outputs.append(distance_column)
    if include_pca:
        pca_x_column = _unique_name(result.columns, f"{cluster_column}_PCA1")
        result[pca_x_column] = np.nan
        result.loc[valid_index, pca_x_column] = projection[:, 0]
        pca_y_column = _unique_name(result.columns, f"{cluster_column}_PCA2")
        result[pca_y_column] = np.nan
        result.loc[valid_index, pca_y_column] = projection[:, 1]
        outputs.extend([pca_x_column, pca_y_column])

    metrics = _quality_metrics(matrix, labels, fitted.inertia_, int(random_state))
    diagnostics = _diagnostics(matrix, algorithm, int(random_state))
    profile = pd.DataFrame(matrix, columns=used).assign(_cluster=labels + 1)
    profile = profile.groupby("_cluster", sort=True).mean()
    projection_idx = _sample_indices(len(matrix), MAX_PROJECTION_POINTS, int(random_state))
    sizes = pd.Series(labels + 1).value_counts().sort_index()

    analysis = {
        "features": used,
        "constant_features": constant,
        "input_rows": int(len(frame)),
        "used_rows": int(len(matrix)),
        "excluded_rows": int(len(frame) - len(matrix)),
        "k": k,
        "actual_k": int(actual_clusters),
        "algorithm": algorithm,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "projection": {
            "x": projection[projection_idx, 0].tolist(),
            "y": projection[projection_idx, 1].tolist(),
            "cluster": [f"Кластер {value + 1}" for value in labels[projection_idx]],
            "row": [str(value) for value in valid_index[projection_idx]],
            "explained_variance": projection_model.explained_variance_ratio_.tolist(),
        },
        "sizes": {
            "cluster": [f"Кластер {int(value)}" for value in sizes.index],
            "count": [int(value) for value in sizes.values],
        },
        "profile": {
            "features": used,
            "clusters": [f"Кластер {int(value)}" for value in profile.index],
            "values": profile.to_numpy().tolist(),
        },
        "outputs": outputs,
    }
    committed_step = {
        "type": "clustering",
        "label": f"Кластеризация · {algorithm} · K={k}",
        "inputs": used,
        "outputs": outputs,
        "params": {
            "algorithm": algorithm,
            "k": k,
            "scaling": scaling,
            "missing_policy": missing_policy,
            "random_state": int(random_state),
            "constant_features": constant,
            "metrics": metrics,
        },
    }
    return CachedClusteringResult(result, analysis, committed_step, signature)
