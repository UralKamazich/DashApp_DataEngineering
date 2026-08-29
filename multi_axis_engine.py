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


STATE_VERSION = 2
# ``go.Scatter`` is SVG-based. This is a total budget across all visible
# series, not a per-series allowance that grows dangerously with every axis.
DEFAULT_MAX_VISUAL_POINTS = 20_000

SERIES_TYPES = ("line", "scatter", "line+markers", "step", "area", "box")
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
    "box": "box",
    "boxplot": "box",
    "box-plot": "box",
    "ящиксусами": "box",
}

_BOX_POINT_MODES = {
    "outliers": "outliers",
    "suspectedoutliers": "suspectedoutliers",
    "all": "all",
    "none": False,
    "false": False,
    "off": False,
    "": False,
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


def _boolean(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


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


def _box_points(value: Any) -> str | bool:
    if value is False:
        return False
    if value is None:
        return "outliers"
    return _BOX_POINT_MODES.get(str(value).strip().lower(), "outliers")


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

        color = str(raw.get("color") or DEFAULT_COLORS[ordinal % len(DEFAULT_COLORS)])
        normalized = {
            "id": series_id,
            "y": y_column,
            "type": _series_type(raw.get("type", raw.get("chart_type"))),
            "name": str(raw.get("name") or y_column),
            "color": color,
            "axis_id": axis_id,
            "visible": bool(raw.get("visible", True)),
            "line_width": _finite_float(raw.get("line_width"), 2.0, 0.25, 20.0),
            "line_dash": str(raw.get("line_dash") or "solid"),
            "smooth": _boolean(
                raw.get("smooth", str(raw.get("line_shape") or "").lower() == "spline")
            ),
            "marker_size": _finite_float(raw.get("marker_size"), 7.0, 1.0, 100.0),
            "opacity": _finite_float(raw.get("opacity"), 1.0, 0.05, 1.0),
            "fill_opacity": _finite_float(raw.get("fill_opacity"), 0.22, 0.02, 1.0),
            "box_points": _box_points(raw.get("box_points")),
            "step_shape": str(raw.get("step_shape") or "hv"),
            "available": (
                True
                if available is None
                else y_column in available
                and (shared_x is None or shared_x in available)
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


def required_columns(state: Mapping[str, Any] | None, *, visible_only: bool = True) -> list[Any]:
    """Return de-duplicated source columns required to draw ``state``."""
    normalized = normalize_multi_axis_state(state)
    result: list[Any] = []
    for series in normalized["series"]:
        if visible_only and not series["visible"]:
            continue
        for column in (normalized["shared_x"], series["y"]):
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
    x_labels: Any | None = None,
    box_width: float | None = None,
) -> go.BaseTraceType:
    chart_type = series["type"]
    if chart_type == "box":
        kwargs: dict[str, Any] = {
            "y": y,
            "name": series["name"],
            "yaxis": yaxis,
            "boxpoints": series["box_points"],
            "line": {"color": series["color"], "width": series["line_width"]},
            "marker": {"color": series["color"], "size": series["marker_size"]},
            # Keep the standard Plotly box appearance. Visibility between
            # overlaying Y-axis layers is handled by the transparent plotting
            # surface rather than by removing the box body itself.
            "fillcolor": _rgba(series["color"], 0.24),
            "opacity": 1.0,
            "alignmentgroup": "multi-y-boxes",
            "offsetgroup": str(series["id"]),
            "meta": {"series_id": series["id"], "axis_id": series["axis_id"]},
        }
        if x is not None:
            kwargs["x"] = x
        if x_labels is not None:
            kwargs.update(
                customdata=x_labels,
                hovertemplate="%{customdata}<br>%{y}<extra>%{fullData.name}</extra>",
            )
        if box_width is not None:
            kwargs["width"] = box_width
        return go.Box(**kwargs)

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
            "hv", "vh", "hvh", "vhv", "linear",
        } else "hv"
    elif chart_type in {"line", "line+markers", "area"} and series["smooth"]:
        line.update(shape="spline", smoothing=0.7)
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
    needs_svg_spline = (
        chart_type in {"line", "line+markers", "area"} and series["smooth"]
    )
    trace_class = go.Scatter if render_mode == "svg" or needs_svg_spline else go.Scattergl
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
        x_column = shared_x
        needed = [series["y"]] + ([x_column] if x_column is not None else [])
        missing = [column for column in needed if column not in columns]
        for column in missing:
            if column not in missing_columns:
                missing_columns.append(column)
        if not missing:
            valid_series.append(series)

    selected_columns: list[Any] = []
    for series in valid_series:
        for column in (shared_x, series["y"]):
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
    box_series = [series for series in valid_series if series["type"] == "box"]
    grouped_box_x = shared_x is not None and bool(box_series)
    grouped_x_positions = None
    grouped_x_labels = None
    grouped_x_categories: list[Any] = []
    box_offsets: dict[str, float] = {}
    box_width = None
    if grouped_box_x:
        grouped_x_labels = visual[shared_x]
        codes, categories = pd.factorize(grouped_x_labels, sort=False)
        grouped_x_positions = codes.astype(float)
        grouped_x_positions[codes < 0] = np.nan
        grouped_x_categories = categories.tolist()
        slot = 0.72 / len(box_series)
        box_width = slot * 0.82
        midpoint = (len(box_series) - 1) / 2
        box_offsets = {
            str(series["id"]): (index - midpoint) * slot
            for index, series in enumerate(box_series)
        }
    for series in valid_series:
        x_column = shared_x
        if grouped_box_x:
            x_values = grouped_x_positions
            if series["type"] == "box":
                x_values = grouped_x_positions + box_offsets[str(series["id"])]
        else:
            x_values = visual[x_column] if x_column is not None else (
                None if series["type"] == "box" else source_positions
            )
        trace_axis, _ = plotly_axis_refs[series["axis_id"]]
        figure.add_trace(_trace_for_series(
            series,
            x=x_values,
            y=visual[series["y"]],
            yaxis=trace_axis,
            render_mode=normalized_render_mode,
            x_labels=grouped_x_labels if grouped_box_x else None,
            box_width=box_width if grouped_box_x and series["type"] == "box" else None,
        ))

    series_by_axis: dict[str, list[dict[str, Any]]] = {}
    for series in valid_series:
        series_by_axis.setdefault(series["axis_id"], []).append(series)
    left_axes = sum(axis_by_id[axis_id]["side"] == "left" for axis_id in used_axis_ids)
    right_axes = len(used_axis_ids) - left_axes
    max_extra_axes = max(max(0, left_axes - 1), max(0, right_axes - 1), 1)
    axis_position_step = min(0.06, 0.30 / max_extra_axes)
    x_domain_left = max(0, left_axes - 1) * axis_position_step
    x_domain_right = 1.0 - max(0, right_axes - 1) * axis_position_step
    axis_layout: dict[str, Any] = {}
    side_axis_counts = {"left": 0, "right": 0}
    for index, axis_id in enumerate(used_axis_ids):
        axis = axis_by_id[axis_id]
        side = axis["side"]
        side_ordinal = side_axis_counts[side]
        side_axis_counts[side] += 1
        related_series = series_by_axis[axis_id]
        # A private axis always follows its data pair. A shared scale may use
        # an explicitly chosen neutral color.
        color = related_series[0]["color"] if len(related_series) == 1 else (
            axis.get("color") or related_series[0]["color"]
        )
        _, layout_key = plotly_axis_refs[axis_id]
        axis_title = related_series[0].get("name") or axis.get("title") or "Y"
        axis_config: dict[str, Any] = {
            "title": {
                "text": axis_title,
                "font": {"color": color},
                "standoff": 5,
            },
            "tickfont": {"color": color},
            "linecolor": color,
            "tickcolor": color,
            "showline": True,
            "ticks": "outside",
            "side": side,
            "type": axis["type"],
            "visible": axis["visible"],
            "showgrid": index == 0,
            "zeroline": False,
            "automargin": False,
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
        # Reserve a real horizontal strip for every axis. Unlike pixel
        # ``shift``, an explicit free-axis position moves the line, ticks and
        # title as one unit and keeps them outside the data domain.
        position = (
            x_domain_left - side_ordinal * axis_position_step
            if side == "left"
            else x_domain_right + side_ordinal * axis_position_step
        )
        axis_config.update(
            overlaying="y",
            anchor="free",
            position=max(0.0, min(1.0, position)),
            autoshift=False,
            shift=0,
        )
        axis_layout[layout_key] = axis_config

    x_title = str(shared_x) if shared_x is not None else "Номер строки"

    xaxis_config: dict[str, Any] = {
        "title": {"text": x_title},
        "domain": [x_domain_left, x_domain_right],
        "automargin": True,
        "showline": True,
        "ticks": "outside",
        "ticklen": 5,
        "tickwidth": 1,
        "zeroline": False,
    }
    if grouped_box_x:
        xaxis_config.update(
            tickmode="array",
            tickvals=list(range(len(grouped_x_categories))),
            ticktext=[str(value) for value in grouped_x_categories],
            range=[-0.55, max(0.55, len(grouped_x_categories) - 0.45)],
        )

    figure.update_layout(
        template=template or "plotly",
        title={"text": normalized["title"] or None},
        height=normalized["height"],
        width=normalized["width"],
        autosize=normalized["width"] is None,
        showlegend=normalized["show_legend"],
        hovermode="x unified",
        boxmode="group",
        # Plotly creates overlaying subplot layers for independent Y axes.
        # Their backgrounds must not conceal traces rendered on earlier axes.
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision=multi_axis_uirevision(normalized),
        xaxis=xaxis_config,
        # All visible series axes are y2+ and overlay this stable, hidden
        # carrier. Removing/reordering another series therefore never changes
        # an existing trace's Plotly axis reference.
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
        margin={
            "l": 60,
            "r": 45,
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
