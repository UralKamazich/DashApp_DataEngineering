# -*- coding: utf-8 -*-
"""Pure state and figure engine for a reusable multi-Y-axis workspace.

The module deliberately has no Dash dependencies.  A page or a future
dashboard widget can keep the normalized state in its own store and call
``build_multi_axis_figure`` with any pandas frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


STATE_VERSION = 1
# ``go.Scatter`` is SVG-based. This is a total budget across all visible
# series, not a per-series allowance that grows dangerously with every axis.
DEFAULT_MAX_VISUAL_POINTS = 20_000

SERIES_TYPES = ("line", "scatter", "line+markers", "step", "area")
AXIS_SIDES = ("left", "right")

DEFAULT_COLORS = (
    "#228be6",
    "#fa5252",
    "#40c057",
    "#7950f2",
    "#fd7e14",
    "#15aabf",
    "#e64980",
    "#82c91e",
    "#be4bdb",
    "#fab005",
)

_TYPE_ALIASES = {
    "line": "line",
    "lines": "line",
    "линия": "line",
    "scatter": "scatter",
    "markers": "scatter",
    "points": "scatter",
    "точки": "scatter",
    "line+markers": "line+markers",
    "lines+markers": "line+markers",
    "line-markers": "line+markers",
    "линия+точки": "line+markers",
    "step": "step",
    "steps": "step",
    "ступенчатая": "step",
    "area": "area",
    "область": "area",
}


@dataclass(frozen=True)
class MultiAxisBuildResult:
    """Result returned by :func:`build_multi_axis_figure`.

    ``metadata`` is intentionally JSON-compatible so it can be placed into a
    ``dcc.Store`` unchanged.
    """

    figure: go.Figure
    state: dict[str, Any]
    metadata: dict[str, Any]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _column(value: Any) -> Any | None:
    """Return a JSON-friendly column key, treating blank strings as missing."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)


def _finite_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return min(maximum, max(minimum, result))


def _next_id(prefix: str, used: set[str]) -> str:
    number = 1
    while f"{prefix}-{number}" in used:
        number += 1
    result = f"{prefix}-{number}"
    used.add(result)
    return result


def _stable_id(value: Any, prefix: str, used: set[str]) -> str:
    candidate = str(value).strip() if value is not None else ""
    if candidate and candidate not in used:
        used.add(candidate)
        return candidate
    return _next_id(prefix, used)


def _derived_plotly_ref(axis_id: str) -> str:
    """Derive a persistent Plotly ref when a UI store has not saved one yet."""
    prefix, separator, suffix = axis_id.rpartition("-")
    if separator and prefix == "axis" and suffix.isdigit():
        return f"y{int(suffix) + 1}"
    # Browser-created axis IDs contain timestamps/random suffixes. Deriving a
    # ref from that stable ID means deletion/reordering cannot renumber it even
    # before the normalized state has been written back to a Store.
    number = int(hashlib.sha1(axis_id.encode("utf-8")).hexdigest()[:12], 16)
    return f"y{number % 900_000_000 + 2}"


def _stable_plotly_ref(value: Any, used: set[str], *, axis_id: str) -> str:
    """Keep an existing yN reference or allocate a stable carrier-backed one."""
    candidate = str(value or "").strip().lower()
    if candidate.startswith("yaxis"):
        candidate = "y" + candidate[5:]
    if candidate.startswith("y") and candidate[1:].isdigit() and int(candidate[1:]) >= 2:
        if candidate not in used:
            used.add(candidate)
            return candidate
    result = _derived_plotly_ref(axis_id)
    number = int(result[1:])
    while result in used:
        number += 1
        result = f"y{number}"
    used.add(result)
    return result


def _series_type(value: Any) -> str:
    key = str(value or "line").strip().lower().replace(" ", "")
    return _TYPE_ALIASES.get(key, "line")


def _axis_range(axis: Mapping[str, Any]) -> list[float] | None:
    raw_range = axis.get("range")
    if isinstance(raw_range, Sequence) and not isinstance(raw_range, (str, bytes)):
        if len(raw_range) >= 2:
            values = raw_range[:2]
        else:
            values = ()
    else:
        values = (axis.get("min"), axis.get("max"))
    if len(values) != 2 or any(value is None or value == "" for value in values):
        return None
    try:
        result = [float(values[0]), float(values[1])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(value) for value in result) else None


def _normalize_axis(
    raw: Mapping[str, Any],
    *,
    axis_id: str,
    ordinal: int,
    default_title: str,
    default_color: str,
    plotly_ref: str,
    default_side: str | None = None,
) -> dict[str, Any]:
    side = str(raw.get("side") or default_side or ("left" if ordinal % 2 == 0 else "right"))
    side = side.lower() if side.lower() in AXIS_SIDES else "left"
    axis_type = str(raw.get("type", raw.get("scale", "linear"))).lower()
    axis_type = "log" if axis_type in {"log", "logarithmic", "лог", "логарифмическая"} else "linear"
    requested_range = _axis_range(raw)
    autorange = raw.get("autorange", raw.get("range_auto", requested_range is None))
    reversed_axis = bool(raw.get("reversed", False))
    if reversed_axis and bool(autorange):
        normalized_autorange: bool | str = "reversed"
    else:
        normalized_autorange = bool(autorange)

    color_mode = str(raw.get("color_mode") or ("custom" if raw.get("color") else "series"))
    color_mode = "custom" if color_mode == "custom" else "series"
    color = raw.get("color") if color_mode == "custom" else default_color
    return {
        "id": axis_id,
        "plotly_ref": plotly_ref,
        "title": str(raw.get("title") or default_title or "Y"),
        "side": side,
        "type": axis_type,
        "autorange": normalized_autorange,
        "range": requested_range,
        "reversed": reversed_axis,
        "visible": bool(raw.get("visible", True)),
        "color": str(color or default_color),
        "color_mode": color_mode,
        "tickformat": str(raw.get("tickformat") or ""),
    }


def normalize_multi_axis_state(
    state: Mapping[str, Any] | None,
    available_columns: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return a canonical, JSON-compatible workspace state.

    Existing non-empty series and axis IDs are preserved.  Missing or
    duplicate IDs receive deterministic ``series-N``/``axis-N`` values.  A
    series gets its own axis by default; assigning the same ``axis_id`` to
    multiple series intentionally shares that scale.

    ``available_columns`` is advisory: the configuration is retained when a
    dataset lacks one of its columns, while an ``available`` flag lets the UI
    show the stale assignment without destroying it.
    """
    source = _mapping(state)
    shared_x = _column(source.get("shared_x", source.get("x")))
    available = set(available_columns) if available_columns is not None else None

    raw_axes = source.get("axes")
    if not isinstance(raw_axes, Sequence) or isinstance(raw_axes, (str, bytes)):
        raw_axes = []
    axis_inputs: dict[str, dict[str, Any]] = {}
    axis_order: list[str] = []
    used_axis_ids: set[str] = set()
    for raw_axis_value in raw_axes:
        raw_axis = _mapping(raw_axis_value)
        axis_id = _stable_id(raw_axis.get("id"), "axis", used_axis_ids)
        axis_inputs[axis_id] = raw_axis
        axis_order.append(axis_id)

    raw_series = source.get("series")
    if not isinstance(raw_series, Sequence) or isinstance(raw_series, (str, bytes)):
        raw_series = []

    used_series_ids: set[str] = set()
    normalized_series: list[dict[str, Any]] = []
    nested_axis_inputs: dict[str, dict[str, Any]] = {}
    for ordinal, raw_series_value in enumerate(raw_series):
        raw = _mapping(raw_series_value)
        y_column = _column(raw.get("y", raw.get("column")))
        if y_column is None:
            continue

        series_id = _stable_id(raw.get("id"), "series", used_series_ids)
        nested_axis = _mapping(raw.get("axis"))
        explicit_axis = raw.get("axis_id")
        if explicit_axis is None and isinstance(raw.get("axis"), str):
            explicit_axis = raw.get("axis")
        if explicit_axis is None:
            explicit_axis = nested_axis.get("id")
        explicit_axis = str(explicit_axis).strip() if explicit_axis is not None else ""
        if explicit_axis:
            axis_id = explicit_axis
            if axis_id not in used_axis_ids:
                used_axis_ids.add(axis_id)
                axis_order.append(axis_id)
        else:
            axis_id = _next_id("axis", used_axis_ids)
            axis_order.append(axis_id)
        if axis_id not in axis_inputs:
            axis_inputs[axis_id] = {}
        if nested_axis:
            nested_axis_inputs.setdefault(axis_id, {}).update(nested_axis)
        if raw.get("side") and not nested_axis_inputs.get(axis_id, {}).get("side"):
            nested_axis_inputs.setdefault(axis_id, {})["side"] = raw["side"]

        x_mode = str(raw.get("x_mode") or ("individual" if raw.get("x") is not None else "shared"))
        x_mode = "individual" if x_mode.lower() in {"individual", "own", "series"} else "shared"
        own_x = _column(raw.get("x")) if x_mode == "individual" else None
        color = str(raw.get("color") or DEFAULT_COLORS[ordinal % len(DEFAULT_COLORS)])
        normalized = {
            "id": series_id,
            "y": y_column,
            "x_mode": x_mode,
            "x": own_x,
            "type": _series_type(raw.get("type", raw.get("chart_type"))),
            "name": str(raw.get("name") or y_column),
            "color": color,
            "axis_id": axis_id,
            "visible": bool(raw.get("visible", True)),
            "line_width": _finite_float(raw.get("line_width"), 2.0, 0.25, 20.0),
            "line_dash": str(raw.get("line_dash") or "solid"),
            "marker_size": _finite_float(raw.get("marker_size"), 7.0, 1.0, 100.0),
            "opacity": _finite_float(raw.get("opacity"), 1.0, 0.05, 1.0),
            "fill_opacity": _finite_float(raw.get("fill_opacity"), 0.22, 0.02, 1.0),
            "step_shape": str(raw.get("step_shape") or "hv"),
            "available": (
                True
                if available is None
                else y_column in available
                and (x_mode != "individual" or own_x is None or own_x in available)
                and (x_mode != "shared" or shared_x is None or shared_x in available)
            ),
        }
        normalized_series.append(normalized)

    # A series may ask to share the axis of another series without knowing its
    # axis ID.  Resolve it only after every stable series ID has been assigned.
    series_axis_by_id = {item["id"]: item["axis_id"] for item in normalized_series}
    for normalized, raw_series_value in zip(normalized_series, [
        item for item in raw_series if _column(_mapping(item).get("y", _mapping(item).get("column"))) is not None
    ]):
        raw = _mapping(raw_series_value)
        shared_with = str(raw.get("share_axis_with") or "").strip()
        if shared_with in series_axis_by_id:
            normalized["axis_id"] = series_axis_by_id[shared_with]

    first_series_for_axis: dict[str, dict[str, Any]] = {}
    for series in normalized_series:
        first_series_for_axis.setdefault(series["axis_id"], series)
        if series["axis_id"] not in axis_order:
            axis_order.append(series["axis_id"])
            used_axis_ids.add(series["axis_id"])

    normalized_axes: list[dict[str, Any]] = []
    used_plotly_refs: set[str] = set()
    for ordinal, axis_id in enumerate(axis_order):
        related = first_series_for_axis.get(axis_id)
        defaults = related or {
            "name": "Y",
            "y": "Y",
            "color": DEFAULT_COLORS[ordinal % len(DEFAULT_COLORS)],
        }
        merged = dict(nested_axis_inputs.get(axis_id, {}))
        merged.update(axis_inputs.get(axis_id, {}))
        normalized_axes.append(_normalize_axis(
            merged,
            axis_id=axis_id,
            ordinal=ordinal,
            default_title=str(defaults.get("name") or defaults.get("y") or "Y"),
            default_color=str(defaults.get("color") or DEFAULT_COLORS[ordinal % len(DEFAULT_COLORS)]),
            plotly_ref=_stable_plotly_ref(
                merged.get("plotly_ref"), used_plotly_refs, axis_id=axis_id,
            ),
        ))

    # A private axis belongs visually to its one series. Its color must keep
    # following that series even when a previously normalized store contains
    # the old generated axis color.
    related_by_axis: dict[str, list[dict[str, Any]]] = {}
    for series in normalized_series:
        related_by_axis.setdefault(series["axis_id"], []).append(series)
    for axis in normalized_axes:
        related = related_by_axis.get(axis["id"], [])
        if len(related) == 1:
            axis["color"] = related[0]["color"]
            axis["color_mode"] = "series"

    normalized_state = {
        "version": STATE_VERSION,
        "shared_x": shared_x,
        "series": normalized_series,
        "axes": normalized_axes,
        "title": str(source.get("title") or ""),
        "height": int(_finite_float(source.get("height"), 700, 240, 5000)),
        "width": (
            int(_finite_float(source.get("width"), 1000, 320, 8000))
            if source.get("width") not in (None, "")
            else None
        ),
        "show_legend": bool(source.get("show_legend", True)),
        "view_revision": int(_finite_float(source.get("view_revision"), 0, 0, 1_000_000_000)),
    }
    if available is not None:
        normalized_state["shared_x_available"] = shared_x is None or shared_x in available
    return normalized_state


def _effective_x(series: Mapping[str, Any], shared_x: Any | None) -> Any | None:
    if series.get("x_mode") == "individual" and series.get("x") is not None:
        return series.get("x")
    return shared_x


def required_columns(state: Mapping[str, Any] | None, *, visible_only: bool = True) -> list[Any]:
    """Return de-duplicated source columns required to draw ``state``."""
    normalized = normalize_multi_axis_state(state)
    result: list[Any] = []
    for series in normalized["series"]:
        if visible_only and not series["visible"]:
            continue
        for column in (_effective_x(series, normalized["shared_x"]), series["y"]):
            if column is not None and column not in result:
                result.append(column)
    return result


def multi_axis_uirevision(state: Mapping[str, Any] | None) -> str:
    """Return a key that changes only with the graph coordinate system.

    Names, colors, widths, marker sizes, opacity, trace rendering type and axis
    side are cosmetic and therefore deliberately absent from the signature.
    """
    normalized = normalize_multi_axis_state(state)
    axis_by_id = {axis["id"]: axis for axis in normalized["axes"]}
    structural = {
        "shared_x": normalized["shared_x"],
        "series": [
            {
                "id": series["id"],
                "x": _effective_x(series, normalized["shared_x"]),
                "y": series["y"],
                "axis": series["axis_id"],
            }
            for series in normalized["series"]
        ],
        "axes": [
            {
                "id": axis_id,
                "plotly_ref": axis_by_id.get(axis_id, {}).get("plotly_ref"),
                "type": axis_by_id.get(axis_id, {}).get("type", "linear"),
                "autorange": axis_by_id.get(axis_id, {}).get("autorange", True),
                "range": axis_by_id.get(axis_id, {}).get("range"),
            }
            for axis_id in dict.fromkeys(series["axis_id"] for series in normalized["series"])
        ],
        "view_revision": normalized["view_revision"],
    }
    payload = json.dumps(structural, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "multi-y:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _rgba(color: str, opacity: float) -> str:
    value = color.strip()
    if value.startswith("#") and len(value) in {4, 7}:
        if len(value) == 4:
            value = "#" + "".join(character * 2 for character in value[1:])
        try:
            red, green, blue = (int(value[index:index + 2], 16) for index in (1, 3, 5))
            return f"rgba({red},{green},{blue},{opacity:.3f})"
        except ValueError:
            pass
    return color


def _sample_positions(row_count: int, row_limit: int) -> tuple[np.ndarray | slice, bool]:
    if row_count <= row_limit:
        return slice(None), False
    if row_limit <= 1:
        return np.array([0], dtype=np.int64), True
    # Stable, ordered and endpoint-preserving decimation is preferable for a
    # workspace dominated by line/step signals.
    positions = np.linspace(0, row_count - 1, row_limit, dtype=np.int64)
    return positions, True


def _trace_for_series(
    series: Mapping[str, Any],
    *,
    x: Any,
    y: Any,
    yaxis: str,
    render_mode: str,
) -> go.BaseTraceType:
    chart_type = series["type"]
    mode = {
        "line": "lines",
        "scatter": "markers",
        "line+markers": "lines+markers",
        "step": "lines",
        "area": "lines",
    }[chart_type]
    line = {
        "color": series["color"],
        "width": series["line_width"],
        "dash": series["line_dash"],
    }
    if chart_type == "step":
        line["shape"] = series["step_shape"] if series["step_shape"] in {
            "hv", "vh", "hvh", "vhv", "linear", "spline",
        } else "hv"
    kwargs: dict[str, Any] = {
        "x": x,
        "y": y,
        "name": series["name"],
        "mode": mode,
        "yaxis": yaxis,
        "opacity": series["opacity"],
        "line": line,
        "marker": {"color": series["color"], "size": series["marker_size"]},
        "connectgaps": False,
        "meta": {"series_id": series["id"], "axis_id": series["axis_id"]},
    }
    if chart_type == "area":
        kwargs.update(fill="tozeroy", fillcolor=_rgba(series["color"], series["fill_opacity"]))
    trace_class = go.Scatter if render_mode == "svg" else go.Scattergl
    return trace_class(**kwargs)


def build_multi_axis_figure(
    frame: pd.DataFrame,
    state: Mapping[str, Any] | None,
    *,
    template: str = "plotly",
    render_mode: str = "hybrid",
    max_visual_points: int = DEFAULT_MAX_VISUAL_POINTS,
) -> MultiAxisBuildResult:
    """Build a renderer-safe Plotly multi-axis figure.

    The visual point budget is shared by all visible, valid series.  Sampling
    affects only the temporary frame passed to Plotly; the caller's dataframe
    is neither modified nor widened/copied.  Only required X/Y columns are
    selected.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    try:
        point_budget = int(max_visual_points)
    except (TypeError, ValueError) as error:
        raise ValueError("max_visual_points must be a positive integer") from error
    if point_budget <= 0:
        raise ValueError("max_visual_points must be a positive integer")
    normalized_render_mode = "svg" if str(render_mode).lower() == "svg" else "hybrid"

    normalized = normalize_multi_axis_state(state, frame.columns.tolist())
    shared_x = normalized["shared_x"]
    columns = set(frame.columns.tolist())
    valid_series: list[dict[str, Any]] = []
    missing_columns: list[Any] = []
    for series in normalized["series"]:
        if not series["visible"]:
            continue
        x_column = _effective_x(series, shared_x)
        needed = [series["y"]] + ([x_column] if x_column is not None else [])
        missing = [column for column in needed if column not in columns]
        for column in missing:
            if column not in missing_columns:
                missing_columns.append(column)
        if not missing:
            valid_series.append(series)

    selected_columns: list[Any] = []
    for series in valid_series:
        for column in (_effective_x(series, shared_x), series["y"]):
            if column is not None and column not in selected_columns:
                selected_columns.append(column)

    row_count = int(len(frame))
    series_count = len(valid_series)
    row_limit = row_count if not series_count else max(1, point_budget // series_count)
    positions, sampled = _sample_positions(row_count, row_limit)

    if selected_columns:
        # One combined iloc operation avoids ever materializing unrelated
        # columns from a wide source frame.
        column_positions = [frame.columns.get_loc(column) for column in selected_columns]
        if not all(isinstance(position, (int, np.integer)) for position in column_positions):
            raise ValueError("Multi-axis plots require unique dataframe column names")
        visual = frame.iloc[positions, column_positions].copy(deep=False)
    else:
        visual = pd.DataFrame(index=frame.index[positions])
    displayed_rows = int(len(visual))
    if isinstance(positions, slice):
        source_positions = np.arange(row_count, dtype=np.int64)
    else:
        source_positions = positions

    used_axis_ids = list(dict.fromkeys(series["axis_id"] for series in valid_series))
    axis_by_id = {axis["id"]: axis for axis in normalized["axes"]}
    plotly_axis_refs: dict[str, tuple[str, str]] = {}
    for axis_id in used_axis_ids:
        trace_ref = axis_by_id[axis_id]["plotly_ref"]
        plotly_axis_refs[axis_id] = (trace_ref, f"yaxis{trace_ref[1:]}")

    figure = go.Figure()
    for series in valid_series:
        x_column = _effective_x(series, shared_x)
        x_values = visual[x_column] if x_column is not None else source_positions
        trace_axis, _ = plotly_axis_refs[series["axis_id"]]
        figure.add_trace(_trace_for_series(
            series,
            x=x_values,
            y=visual[series["y"]],
            yaxis=trace_axis,
            render_mode=normalized_render_mode,
        ))

    series_by_axis: dict[str, list[dict[str, Any]]] = {}
    for series in valid_series:
        series_by_axis.setdefault(series["axis_id"], []).append(series)
    axis_layout: dict[str, Any] = {}
    for index, axis_id in enumerate(used_axis_ids):
        axis = axis_by_id[axis_id]
        related_series = series_by_axis[axis_id]
        # A private axis always follows its data pair. A shared scale may use
        # an explicitly chosen neutral color.
        color = related_series[0]["color"] if len(related_series) == 1 else (
            axis.get("color") or related_series[0]["color"]
        )
        _, layout_key = plotly_axis_refs[axis_id]
        axis_config: dict[str, Any] = {
            "title": {"text": axis["title"], "font": {"color": color}},
            "tickfont": {"color": color},
            "linecolor": color,
            "tickcolor": color,
            "showline": True,
            "ticks": "outside",
            "side": axis["side"],
            "type": axis["type"],
            "visible": axis["visible"],
            "showgrid": index == 0,
            "zeroline": False,
            "automargin": True,
            "layer": "above traces",
        }
        if axis["tickformat"]:
            axis_config["tickformat"] = axis["tickformat"]
        manual_range = axis["range"] if axis["autorange"] is False else None
        if manual_range is not None and axis["type"] == "log":
            # The workspace accepts real data values (for example 1..1000),
            # whereas Plotly's log-axis layout expects log10 exponents.
            manual_range = (
                [math.log10(value) for value in manual_range]
                if all(value > 0 for value in manual_range)
                else None
            )
        if manual_range is not None:
            axis_config.update(range=manual_range, autorange=False)
        else:
            axis_config["autorange"] = (
                axis["autorange"] if axis["autorange"] is not False else True
            )
        axis_config.update(overlaying="y", anchor="free", autoshift=True)
        axis_layout[layout_key] = axis_config

    effective_x = list(dict.fromkeys(
        _effective_x(series, shared_x) for series in valid_series
    ))
    if not effective_x or effective_x == [None]:
        x_title = "Номер строки"
    elif len(effective_x) == 1:
        x_title = str(effective_x[0])
    else:
        x_title = "X (индивидуальный)"

    left_axes = sum(axis_by_id[axis_id]["side"] == "left" for axis_id in used_axis_ids)
    right_axes = len(used_axis_ids) - left_axes
    figure.update_layout(
        template=template or "plotly",
        title={"text": normalized["title"] or None},
        height=normalized["height"],
        width=normalized["width"],
        autosize=normalized["width"] is None,
        showlegend=normalized["show_legend"],
        hovermode="x unified" if len(effective_x) == 1 else "closest",
        uirevision=multi_axis_uirevision(normalized),
        xaxis={"title": {"text": x_title}, "automargin": True},
        # All visible series axes are y2+ and overlay this stable, hidden
        # carrier. Removing/reordering another series therefore never changes
        # an existing trace's Plotly axis reference.
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
        margin={
            "l": min(280, 60 + max(0, left_axes - 1) * 42),
            "r": min(280, 35 + max(0, right_axes - 1) * 42),
            "t": 55 if normalized["title"] else 25,
            "b": 55,
        },
        **axis_layout,
    )

    displayed_points = displayed_rows * series_count
    visual_sample_message = None
    if sampled:
        visual_sample_message = (
            f"Отображается {displayed_rows:,} из {row_count:,} строк "
            f"({displayed_points:,} точек для {series_count} серий). "
            "Исходные данные не изменены."
        ).replace(",", " ")
    metadata = {
        "source_rows": row_count,
        "displayed_rows": displayed_rows,
        "series_count": series_count,
        "displayed_points": displayed_points,
        "max_visual_points": point_budget,
        "render_mode": normalized_render_mode,
        "sampled": sampled,
        "sampling_method": "evenly_spaced" if sampled else None,
        "selected_columns": selected_columns,
        "missing_columns": missing_columns,
        "skipped_series": len(normalized["series"]) - series_count,
        "visual_sample_message": visual_sample_message,
        "axis_refs": {
            axis_id: {"trace": trace_ref, "layout": layout_key}
            for axis_id, (trace_ref, layout_key) in plotly_axis_refs.items()
        },
    }
    return MultiAxisBuildResult(figure=figure, state=normalized, metadata=metadata)


__all__ = [
    "AXIS_SIDES",
    "DEFAULT_COLORS",
    "DEFAULT_MAX_VISUAL_POINTS",
    "MultiAxisBuildResult",
    "SERIES_TYPES",
    "build_multi_axis_figure",
    "multi_axis_uirevision",
    "normalize_multi_axis_state",
    "required_columns",
]
