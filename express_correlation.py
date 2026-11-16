# -*- coding: utf-8 -*-
"""Express correlation analysis: rank every numeric channel against one target.

Designed for wide/long datasets:

* Pearson is computed with masked sums over column chunks, so memory stays
  bounded (~64 MB per chunk) and missing values are handled pairwise exactly
  like ``pandas.DataFrame.corr``.
* Spearman ranks are computed per channel; channels whose missing values
  coincide with the target's reuse one shared ranking pass, the rest are
  re-ranked on their joint mask (exact pairwise-complete Spearman).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import special, stats


CHUNK_ELEMENTS = 8_000_000          # ~64 MB of float64 per working block
MAX_SCATTER_POINTS = 50_000
METHODS = {"pearson": "Пирсон", "spearman": "Спирмен"}
_WORKERS = max(1, min(8, (os.cpu_count() or 2)))


def numeric_channels(frame: pd.DataFrame) -> list[str]:
    return [
        str(column) for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


def _as_float(values, exclude_zero: bool = False) -> np.ndarray:
    array = np.array(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=np.float64)
    _clean_block(array, exclude_zero)
    return array


def _clean_block(array: np.ndarray, exclude_zero: bool) -> np.ndarray:
    """In place: non-finite values (and zeros, if requested) become NaN."""
    array[~np.isfinite(array)] = np.nan
    if exclude_zero:
        array[array == 0] = np.nan
    return array


def channels_after(frame: pd.DataFrame, start_after: str | None) -> list[str]:
    """Numeric channels located strictly to the right of *start_after*."""
    channels = numeric_channels(frame)
    if not start_after:
        return channels
    order = [str(column) for column in frame.columns]
    if start_after not in order:
        raise ValueError(f"Столбец «{start_after}» не найден в датасете.")
    allowed = set(order[order.index(start_after) + 1:])
    return [column for column in channels if column in allowed]


def _column_lookup(frame):
    return {str(column): column for column in frame.columns}


def _pearson_block(X: np.ndarray, y: np.ndarray):
    """Pairwise-complete Pearson r and N between y and every column of X."""
    mask = np.isfinite(X) & np.isfinite(y)[:, None]
    n = mask.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        Xm = np.where(mask, X, 0.0)
        Ym = np.where(mask, y[:, None], 0.0)
        mean_x = Xm.sum(axis=0) / n
        mean_y = Ym.sum(axis=0) / n
        dx = np.where(mask, X - mean_x, 0.0)
        dy = np.where(mask, y[:, None] - mean_y, 0.0)
        sxy = (dx * dy).sum(axis=0)
        sxx = (dx * dx).sum(axis=0)
        syy = (dy * dy).sum(axis=0)
        r = sxy / np.sqrt(sxx * syy)
    r[(sxx <= 0) | (syy <= 0)] = np.nan
    return np.clip(r, -1.0, 1.0), n.astype(np.int64)


def _pearson_pair(x: np.ndarray, y: np.ndarray):
    r, n = _pearson_block(x[:, None], y)
    return float(r[0]), int(n[0])


def _p_values(r: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Two-sided p-value of the t-test for a correlation coefficient."""
    r = np.asarray(r, dtype=np.float64)
    df = np.asarray(n, dtype=np.float64) - 2.0
    p = np.full(r.shape, np.nan)
    ok = np.isfinite(r) & (df > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t2 = df[ok] * r[ok] ** 2 / np.maximum(1.0 - r[ok] ** 2, 1e-300)
        # P(|T| > t) = I_{df/(df+t^2)}(df/2, 1/2)
        p[ok] = special.betainc(df[ok] / 2.0, 0.5, df[ok] / (df[ok] + t2))
    return p


def _iter_blocks(columns, n_rows):
    size = max(1, int(CHUNK_ELEMENTS // max(1, n_rows)))
    for start in range(0, len(columns), size):
        yield columns[start:start + size]


def rank_correlations(frame: pd.DataFrame, target: str, method: str = "pearson",
                      min_periods: int = 10, *, exclude_zero: bool = True,
                      start_after: str | None = None) -> dict:
    """Correlate *target* with every other numeric channel, strongest first.

    ``exclude_zero`` treats zeros as missing values (in the target and every
    channel); ``start_after`` limits the analysis to columns located to the
    right of that column.
    """
    method = method if method in METHODS else "pearson"
    min_periods = max(int(min_periods or 3), 3)
    lookup = _column_lookup(frame)
    if target not in lookup:
        raise ValueError(f"Канал «{target}» не найден в датасете.")

    channels = [column for column in channels_after(frame, start_after) if column != target]
    y = _as_float(frame[lookup[target]].to_numpy(), exclude_zero)
    valid_y = np.isfinite(y)
    n_rows = len(frame.index)

    r_all = np.full(len(channels), np.nan)
    n_all = np.zeros(len(channels), dtype=np.int64)
    position = {column: index for index, column in enumerate(channels)}

    if method == "pearson":
        for block in _iter_blocks(channels, n_rows):
            X = frame[[lookup[c] for c in block]].to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
            _clean_block(X, exclude_zero)
            r, n = _pearson_block(X, y)
            idx = [position[c] for c in block]
            r_all[idx], n_all[idx] = r, n
    else:
        y_sub = y[valid_y]
        y_rank = stats.rankdata(y_sub)
        y_centered = y_rank - y_rank.mean()
        y_norm = float(np.sqrt(np.dot(y_centered, y_centered)))

        def spearman_column(x):
            joint = np.isfinite(x)
            count = int(joint.sum())
            if count < 3:
                return math.nan, count
            if count == x.size:
                rx = stats.rankdata(x)
                rx -= rx.mean()
                denominator = float(np.sqrt(np.dot(rx, rx))) * y_norm
                if denominator <= 0:
                    return math.nan, count
                return float(np.dot(rx, y_centered) / denominator), count
            r, _ = _pearson_pair(stats.rankdata(x[joint]), stats.rankdata(y_sub[joint]))
            return r, count

        sub_rows = max(1, int(valid_y.sum()))
        # numpy sorting releases the GIL, so ranking parallelises across cores.
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            for block in _iter_blocks(channels, sub_rows):
                X = frame[[lookup[c] for c in block]].to_numpy(
                    dtype=np.float64, na_value=np.nan, copy=True
                )[valid_y]
                _clean_block(X, exclude_zero)
                results = pool.map(spearman_column, (X[:, i] for i in range(X.shape[1])))
                for column, (r, count) in zip(block, results):
                    idx = position[column]
                    r_all[idx], n_all[idx] = r, count

    p_all = _p_values(r_all, n_all)
    rows = []
    excluded = 0
    for column, r, n, p in zip(channels, r_all, n_all, p_all):
        if not np.isfinite(r) or n < min_periods:
            excluded += 1
            continue
        rows.append({
            "channel": column,
            "r": round(float(r), 6),
            "abs_r": round(abs(float(r)), 6),
            "n": int(n),
            "p": float(p) if np.isfinite(p) else None,
        })
    rows.sort(key=lambda row: row["abs_r"], reverse=True)
    return {
        "target": target,
        "method": method,
        "rows": rows,
        "channels_total": len(channels),
        "excluded": excluded,
        "row_count": n_rows,
        "target_valid": int(valid_y.sum()),
        "exclude_zero": bool(exclude_zero),
        "start_after": start_after or None,
    }


def crossplot_points(frame: pd.DataFrame, x_column: str, y_column: str,
                     max_points: int = MAX_SCATTER_POINTS, seed: int = 0,
                     *, exclude_zero: bool = True) -> dict:
    """Paired finite points for a scatter plot, randomly thinned if large."""
    lookup = _column_lookup(frame)
    for column in (x_column, y_column):
        if column not in lookup:
            raise ValueError(f"Канал «{column}» не найден в датасете.")
    x = _as_float(frame[lookup[x_column]].to_numpy(), exclude_zero)
    y = _as_float(frame[lookup[y_column]].to_numpy(), exclude_zero)
    joint = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    total = int(joint.size)
    pearson, _ = _pearson_pair(x[joint], y[joint]) if total >= 3 else (math.nan, 0)
    spearman = math.nan
    if total >= 3:
        spearman, _ = _pearson_pair(stats.rankdata(x[joint]), stats.rankdata(y[joint]))
    if total > max_points:
        rng = np.random.default_rng(seed)
        joint = np.sort(rng.choice(joint, size=max_points, replace=False))

    fit = None
    if total >= 3:
        xs, ys = x[joint], y[joint]
        if np.ptp(xs) > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            x0, x1 = float(np.min(xs)), float(np.max(xs))
            fit = {"x": [x0, x1], "y": [slope * x0 + intercept, slope * x1 + intercept],
                   "slope": float(slope), "intercept": float(intercept)}

    def _clean(value):
        return None if value is None or not math.isfinite(value) else round(float(value), 6)

    return {
        "x_column": x_column,
        "y_column": y_column,
        "x": x[joint].tolist(),
        "y": y[joint].tolist(),
        "total": total,
        "shown": int(joint.size),
        "pearson": _clean(pearson),
        "spearman": _clean(spearman),
        "fit": fit,
    }
