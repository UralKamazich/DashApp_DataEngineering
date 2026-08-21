# -*- coding: utf-8 -*-
"""
Callbacks: конвейер данных — фильтрация, биннинг, кластеризация, агрегаты.

Разделено на два независимых колбэка:
  1. apply_filters — stored-data + применённые фильтры → filtered-data
  2. run_de_operations — DE-кнопки + filtered-data → filtered-data + новые колонки
"""

import re
import logging
import pandas as pd
import numpy as np
from dash import callback, Output, Input, State, no_update, ALL
from dash.exceptions import PreventUpdate
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from dash_app import app
from utils import apply_filter_conditions, read_df_from_store, meta_from_df, _make_error_notif
from callbacks.filters import _clean_filter_state

logger = logging.getLogger(__name__)


# =========================
# Фильтрация: stored-data + применённые фильтры → filtered-data
# =========================
@app.callback(
    Output("filtered-data", "data", allow_duplicate=True),
    Output("meta-columns", "data", allow_duplicate=True),
    Input("filters-applied-state", "data"),
    Input("filter-applied-logic", "data"),
    State("stored-data", "data"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def apply_filters(filters_state, logic_mode, stored_json, meta_state):
    """Применяет фильтры к исходному датасету → filtered-data (чистый, без DE-колонок)."""
    if not stored_json:
        raise PreventUpdate

    try:
        df = read_df_from_store(stored_json, meta_state)
    except Exception:
        raise PreventUpdate

    if df is None or df.empty:
        raise PreventUpdate

    fs = _clean_filter_state(filters_state)
    meta0 = meta_from_df(df)

    df = apply_filter_conditions(df, fs, meta0, logic_mode or "and")

    js = df.to_json(date_format='iso', orient='split')
    meta = meta_from_df(df)
    return js, meta


# =========================
# Кластерные метрики (elbow/silhouette)
# =========================
@app.callback(
    Output("cluster-metrics", "data", allow_duplicate=True),
    Input("filtered-data", "data"),
    State("cluster-cols", "value"),
    State("meta-columns", "data"),
    prevent_initial_call=True
)
def compute_cluster_metrics(filtered_json, cluster_cols, meta):
    if not filtered_json or not cluster_cols or len(cluster_cols) < 2:
        raise PreventUpdate
    try:
        df = read_df_from_store(filtered_json, meta)
    except Exception:
        raise PreventUpdate
    use = [c for c in cluster_cols if c in df.columns]
    if len(use) < 2:
        raise PreventUpdate

    X = df[use].apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 5:
        raise PreventUpdate

    Xs = StandardScaler().fit_transform(X.values)
    n = Xs.shape[0]
    k_max = max(3, min(12, n - 1))
    ks = list(range(2, k_max + 1))

    inertias = []
    silhouettes = []
    for k in ks:
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
            inertias.append(float(km.inertia_))
            sil = silhouette_score(Xs, km.labels_) if n > k else float("nan")
            silhouettes.append(float(sil))
        except Exception:
            inertias.append(float("nan"))
            silhouettes.append(float("nan"))

    return {"ks": ks, "inertias": inertias, "silhouettes": silhouettes}


# =========================
# DE-операции: биннинг, кластеризация (KMeans), агрегаты
# =========================
@app.callback(
    Output("filtered-data", "data", allow_duplicate=True),
    Output("meta-columns", "data", allow_duplicate=True),
    Output("cluster-metrics", "data", allow_duplicate=True),
    Output("notifications-container", "sendNotifications", allow_duplicate=True),
    Output("de-agg-status", "children", allow_duplicate=True),

    Input("btn-grouping", "n_clicks"),
    Input("btn-cluster",  "n_clicks"),
    Input("btn-agg",      "n_clicks"),

    State("bin-column", "value"),
    State("bin-method", "value"),
    State("bin-k", "value"),
    State("cluster-cols", "value"),
    State("cluster-k", "value"),

    State("agg-keys", "value"),
    State("agg-cols", "value"),
    State("agg-metrics", "value"),
    State("agg-exclude-zeros", "checked"),
    State("agg-exclude-empty", "checked"),

    State("filtered-data", "data"),
    State("meta-columns", "data"),

    prevent_initial_call=True
)
def run_de_operations(
    n_group_btn, n_cluster_btn, n_agg_btn,
    bin_col, bin_method, bin_k,
    cluster_cols, cluster_k,
    agg_keys, agg_cols, agg_metrics, agg_exclude_zeros, agg_exclude_empty,
    filtered_json, meta_state
):
    """Биннинг, кластеризация, агрегаты: добавляют новые колонки в filtered-data."""
    notifications = []
    status_msg = no_update

    if not filtered_json:
        raise PreventUpdate

    try:
        df = read_df_from_store(filtered_json, meta_state)
    except Exception:
        return (
            no_update, no_update, no_update,
            _make_error_notif("Data Engineering: не удалось прочитать датасет."),
            no_update,
        )

    if df is None or df.empty:
        return (
            no_update, no_update, no_update,
            _make_error_notif("Data Engineering: датасет пуст."),
            no_update,
        )

    trig = _ctx.triggered_id

    # --- БИННИНГ ---
    if trig == "btn-grouping":
        if bin_col and (bin_col in df.columns) and (bin_k is not None) and (int(bin_k) >= 2):
            for c in list(df.columns):
                if isinstance(c, str) and c.startswith("Группа(") and c.endswith(")"):
                    df.drop(columns=[c], inplace=True, errors="ignore")
            ser = pd.to_numeric(df[bin_col], errors="coerce")
            valid = ser.dropna()
            if not valid.empty:
                try:
                    if bin_method == "width":
                        cats = pd.cut(valid, bins=int(bin_k), include_lowest=True, duplicates="drop")
                    else:
                        cats = pd.qcut(valid, q=int(bin_k), duplicates="drop")
                    grp_name = f"Группа({bin_col})"
                    df.loc[valid.index, grp_name] = cats.astype("string")
                except Exception as e:
                    notifications.append(_make_error_notif(f"Ошибка биннинга: {e}")["message"] if isinstance(_make_error_notif(f"Ошибка биннинга: {e}"), dict) else str(e))

    # --- КЛАСТЕРИЗАЦИЯ ---
    if trig == "btn-cluster":
        if cluster_cols and len(cluster_cols) >= 2 and (cluster_k or 0) >= 2:
            use_cols = [c for c in cluster_cols if c in df.columns]
            if len(use_cols) >= 2:
                for col in ("PCA1", "PCA2", "Кластеры"):
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True, errors="ignore")
                num_df = df[use_cols].select_dtypes(include=[np.number]).dropna(how='any')
                if num_df.shape[0] >= 3 and num_df.shape[1] >= 2:
                    Xs = StandardScaler().fit_transform(num_df.values)
                    km = KMeans(n_clusters=cluster_k, n_init=10, random_state=42).fit(Xs)
                    labels = pd.Series([f"Кластер {i}" for i in km.labels_], index=num_df.index, dtype="string")
                    df.loc[num_df.index, "Кластеры"] = labels
                    try:
                        pca = PCA(n_components=2).fit_transform(Xs)
                        pca_df = pd.DataFrame(pca, index=num_df.index, columns=["PCA1", "PCA2"])
                        df = df.join(pca_df, how="left")
                    except Exception:
                        pass

    # --- АГРЕГАТЫ ---
    if trig == "btn-agg":
        status_msg = "Data Engineering: ожидает параметров…"
        if not (agg_keys and agg_cols and agg_metrics):
            notifications.append({
                "id": "de-agg-missing",
                "title": "Data Engineering",
                "message": "Выберите ключ(и), столбцы и метрики, затем нажмите Рассчитать.",
                "color": "orange",
                "action": "show",
                "autoClose": 6000,
            })
            status_msg = "Выберите ключ(и), столбцы и метрики."
        else:
            def _norm_token(x) -> str:
                return re.sub(r"[\s\u00A0]+", "", str(x)).lower()

            _col_map = {_norm_token(c): c for c in df.columns}

            def _resolve_columns(selected):
                out = []
                for v in (selected or []):
                    if v in df.columns:
                        out.append(v)
                        continue
                    nv = _norm_token(v)
                    if nv in _col_map:
                        out.append(_col_map[nv])
                return list(dict.fromkeys(out))

            keys = _resolve_columns(agg_keys)
            cols = _resolve_columns(agg_cols)

            _metric_map = {
                "mean": "mean", "avg": "mean", "average": "mean", "среднее": "mean",
                "median": "median", "медиана": "median",
                "mode": "mode", "мода": "mode",
                "sum": "sum", "сумма": "sum",
                "cumsum": "cumsum", "cumulative": "cumsum", "кумулятивнаясумма": "cumsum", "накопительнаясумма": "cumsum",
                "min": "min", "мин": "min",
                "max": "max", "макс": "max",
                "std": "std", "stdev": "std", "sigma": "std", "стандартноеотклонение": "std",
                "count": "count", "количество": "count",
                "nunique": "nunique", "уникальных": "nunique", "числоуникальных": "nunique",
            }
            metrics = []
            for m in (agg_metrics or []):
                nm = _norm_token(m)
                nm = _metric_map.get(nm, nm)
                metrics.append(nm)
            metrics = list(dict.fromkeys([m for m in metrics if m]))

            if not keys or not cols or not metrics:
                notifications.append({
                    "id": "de-agg-invalid",
                    "title": "Data Engineering",
                    "message": "Ключи/столбцы не найдены в текущем датасете или не выбраны метрики.",
                    "color": "orange",
                    "action": "show",
                    "autoClose": 7000,
                })
                status_msg = "Ключи/столбцы отсутствуют в текущем датасете или метрики не выбраны."
            else:
                exclude_zeros = bool(agg_exclude_zeros)
                exclude_empty = bool(agg_exclude_empty)
                added_cols = []
                skipped = []

                def _safe_tag(s: str) -> str:
                    return re.sub(r'[<>:"/\|?*]+', "_", str(s))

                def _make_unique_col(name: str) -> str:
                    base = str(name)
                    if base not in df.columns:
                        return base
                    i = 2
                    while f"{base}_{i}" in df.columns:
                        i += 1
                    return f"{base}_{i}"

                groupers_series = [df[k] for k in keys]

                def _gb_transform(series: pd.Series, func: str):
                    try:
                        return series.groupby(groupers_series, dropna=False, sort=False).transform(func)
                    except TypeError:
                        return series.groupby(groupers_series, sort=False).transform(func)

                def _gb_apply(series: pd.Series, fn):
                    try:
                        return series.groupby(groupers_series, dropna=False, sort=False).transform(fn)
                    except TypeError:
                        return series.groupby(groupers_series, sort=False).transform(fn)

                for col in cols:
                    is_numeric = pd.api.types.is_numeric_dtype(df[col])
                    x = pd.to_numeric(df[col], errors="coerce") if is_numeric else None

                    if is_numeric:
                        x_use = x.copy()
                        if not exclude_empty:
                            x_use = x_use.fillna(0)
                        if exclude_zeros:
                            x_use_no0 = x_use.mask(x_use == 0, np.nan)
                        else:
                            x_use_no0 = x_use

                    for met in metrics:
                        if met in ("mean", "median", "sum", "min", "max", "std"):
                            if not is_numeric:
                                skipped.append((col, met))
                                continue
                            try:
                                res = _gb_transform(x_use_no0, met)
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_{met}")
                            df[out_col] = res
                            added_cols.append(out_col)

                        elif met == "count":
                            try:
                                if is_numeric:
                                    if exclude_empty:
                                        res = _gb_transform(x_use_no0, "count")
                                    else:
                                        res = _gb_apply(df[col], lambda z: len(z))
                                else:
                                    s = df[col]
                                    if exclude_empty:
                                        res = _gb_transform(s, "count")
                                    else:
                                        res = _gb_apply(s, lambda z: len(z))
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_count")
                            df[out_col] = res
                            added_cols.append(out_col)

                        elif met == "nunique":
                            s = df[col]
                            try:
                                if not exclude_empty:
                                    s = s.astype("object").where(pd.notna(s), "<EMPTY>")
                                if exclude_zeros and is_numeric:
                                    s_num = pd.to_numeric(s, errors="coerce")
                                    s = s_num.mask(s_num == 0, np.nan)
                                res = _gb_apply(s, lambda z: z.nunique(dropna=exclude_empty))
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_nunique")
                            df[out_col] = res
                            added_cols.append(out_col)

                        elif met == "mode":
                            s = df[col]
                            try:
                                if not exclude_empty:
                                    s = s.astype("object").where(pd.notna(s), "<EMPTY>")
                                if exclude_zeros and is_numeric:
                                    s_num = pd.to_numeric(s, errors="coerce")
                                    s = s_num.mask(s_num == 0, np.nan)

                                def _mode_first(z: pd.Series):
                                    mm = z.mode(dropna=True)
                                    return mm.iloc[0] if len(mm) else np.nan

                                res = _gb_apply(s, _mode_first)
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_mode")
                            df[out_col] = res
                            added_cols.append(out_col)

                        elif met == "cumsum":
                            if not is_numeric:
                                skipped.append((col, met))
                                continue
                            try:
                                x_cum = x.copy().fillna(0)
                                res = x_cum.groupby(groupers_series, sort=False).cumsum()
                            except Exception:
                                skipped.append((col, met))
                                continue
                            out_col = _make_unique_col(f"{col}_cumsum")
                            df[out_col] = res
                            added_cols.append(out_col)
                        else:
                            skipped.append((col, met))
                            continue

                added_cols = list(dict.fromkeys(added_cols))

                if added_cols:
                    notifications.append({
                        "id": "de-agg-ok",
                        "title": "Data Engineering",
                        "message": f"Добавлены столбцы: {len(added_cols)}",
                        "color": "green",
                        "action": "show",
                        "autoClose": 4000,
                    })
                    status_msg = f"Готово: добавлено {len(added_cols)} столбцов."
                else:
                    notifications.append({
                        "id": "de-agg-none",
                        "title": "Data Engineering",
                        "message": "Новые столбцы не добавлены (возможно, несовместимые метрики или типы данных).",
                        "color": "orange",
                        "action": "show",
                        "autoClose": 7000,
                    })
                    status_msg = "Ничего не добавлено (проверьте типы столбцов и метрики)."

                if skipped:
                    preview = ", ".join([f"{c}:{m}" for c, m in skipped[:5]])
                    more = "" if len(skipped) <= 5 else f" (+{len(skipped)-5})"
                    notifications.append({
                        "id": "de-agg-skip",
                        "title": "Data Engineering",
                        "message": f"Пропущены несовместимые пары (столбец/метрика): {preview}{more}",
                        "color": "yellow",
                        "action": "show",
                        "autoClose": 7000,
                    })

    # Сохраняем результат
    js = df.to_json(date_format='iso', orient='split')
    meta = meta_from_df(df)

    # Кластерные метрики (вычисляем всегда, если есть данные)
    cluster_metrics = no_update
    try:
        if cluster_cols and len(cluster_cols) >= 2:
            X = df[cluster_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if X.shape[0] >= 5:
                Xs = StandardScaler().fit_transform(X.values)
                n = Xs.shape[0]
                k_max = max(3, min(12, n - 1))
                ks = list(range(2, k_max + 1))
                inertias = []
                silhouettes = []
                for k in ks:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
                    inertias.append(float(km.inertia_))
                    sil = silhouette_score(Xs, km.labels_) if n > k else float("nan")
                    silhouettes.append(float(sil))
                cluster_metrics = {"ks": ks, "inertias": inertias, "silhouettes": silhouettes}
    except Exception as e:
        logger.warning(f"[cluster-metrics] fail: {e}")

    return js, meta, cluster_metrics, notifications, status_msg
