# -*- coding: utf-8 -*-
"""Callbacks and the cross-plot data endpoint for «Экспресс корр. анализ»."""

from __future__ import annotations

import time
from threading import Lock

from dash import Input, Output, State, no_update
from flask import jsonify, request

from dash_app import app
from express_correlation import crossplot_points, rank_correlations
from express_correlation_workspace import PAGE_PATH
from lazy_options import lazy_callback
from utils import frame_to_store, is_store_reference, read_df_shared


# Rating calculations on huge datasets are memory-heavy; run one at a time.
_WORK_LOCK = Lock()


@lazy_callback(
    PAGE_PATH,
    "xcorr-options",
    Output("xcorr-target", "options"),
    Output("xcorr-target", "value"),
    Output("xcorr-start-after", "options"),
    Output("xcorr-start-after", "value"),
    Input("filtered-data", "data"),
    State("meta-columns", "data"),
    State("xcorr-target", "value"),
    State("xcorr-start-after", "value"),
    prevent_initial_call=False,
)
def update_express_options(filtered_json, meta, target, start_after):
    meta = meta if isinstance(meta, dict) else {}
    if not filtered_json:
        return [], None, [], None
    columns = [str(column) for column in (meta.get("columns") or [])]
    numeric = [str(column) for column in (meta.get("numeric") or [])]
    target = target if target in numeric else (numeric[0] if numeric else None)
    start_after = start_after if start_after in columns else None
    return numeric, target, columns, start_after


# Correlation needs at least three paired values.
MIN_PAIRS = 3


def build_rating(payload, meta, target, exclude_zero, start_after, min_periods=10):
    """Compute the rating dict sent to the browser (pure, testable)."""
    if not payload:
        return {"error": "Сначала загрузите датасет."}
    if not target:
        return {"error": "Выберите канал."}
    started = time.perf_counter()
    with _WORK_LOCK:
        frame = read_df_shared(payload, meta)
        if frame is None or frame.empty:
            return {"error": "Датасет пуст или больше недоступен — откройте файл заново."}
        result = rank_correlations(
            frame, target,
            min_periods=max(MIN_PAIRS, int(min_periods or MIN_PAIRS)),
            exclude_zero=bool(exclude_zero),
            start_after=start_after or None,
        )
    # The browser fetches cross-plot points by this reference, so plots always
    # use exactly the data the table was computed from.
    result["ref"] = payload if is_store_reference(payload) else frame_to_store(frame)
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def status_text(result):
    if result.get("error"):
        return result["error"]
    parts = [
        f"Пирсон и Спирмен · «{result['target']}»",
        f"каналов: {len(result['rows'])}",
    ]
    if result.get("excluded"):
        parts.append(f"исключено: {result['excluded']}")
    if result.get("start_after"):
        parts.append(f"правее «{result['start_after']}»")
    if result.get("exclude_zero"):
        parts.append("без нулей")
    parts.append(f"строк: {result['row_count']:,}".replace(",", " "))
    parts.append(f"{result['seconds']} с")
    return " · ".join(parts) + " — нажмите на строку, чтобы открыть кросс-плот"


@app.callback(
    Output("xcorr-result", "data"),
    Output("xcorr-status", "children"),
    Input("xcorr-run", "n_clicks"),
    State("filtered-data", "data"),
    State("meta-columns", "data"),
    State("xcorr-target", "value"),
    State("xcorr-exclude-zero", "checked"),
    State("xcorr-start-after", "value"),
    State("xcorr-min-periods", "value"),
    running=[(Output("xcorr-run", "loading"), True, False)],
    prevent_initial_call=True,
)
def calculate_express_rating(n_clicks, payload, meta, target, exclude_zero,
                             start_after, min_periods):
    if not n_clicks:
        return no_update, no_update
    try:
        result = build_rating(
            payload, meta, target, exclude_zero, start_after, min_periods
        )
    except Exception as error:  # surfaced in the table area, not a crash
        result = {"error": f"Не удалось рассчитать: {error}"}
    return result, status_text(result)


app.clientside_callback(
    """
    function(result) {
        if (window.expressCorr) window.expressCorr.render(result);
        return result && result.rows ? result.rows.length : 0;
    }
    """,
    Output("xcorr-rendered", "data"),
    Input("xcorr-result", "data"),
)


@app.server.route("/api/express-corr/crossplot")
def express_crossplot():
    ref = request.args.get("ref", "")
    x_column = request.args.get("x", "")
    y_column = request.args.get("y", "")
    exclude_zero = request.args.get("zero", "1") == "1"
    if not is_store_reference(ref):
        return jsonify({"error": "Нет ссылки на датасет — пересчитайте таблицу."}), 400
    frame = read_df_shared(ref)
    if frame is None or frame.empty:
        return jsonify({"error": "Датасет больше недоступен — пересчитайте таблицу."}), 410
    try:
        return jsonify(crossplot_points(frame, x_column, y_column, exclude_zero=exclude_zero))
    except ValueError as error:
        return jsonify({"error": str(error)}), 404
