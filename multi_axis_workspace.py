# -*- coding: utf-8 -*-
"""Reusable workspace for a Plotly chart with multiple independent Y axes.

``MultiYAxisWorkspace`` deliberately lives beside ``GraphWorkspace`` instead
of extending its chart-type matrix.  A multi-axis chart has a smaller, more
focused contract: one shared X channel and any number of independently styled
Y series, each backed by its own axis by default.

The component owns all of its UI state and uses IDs scoped by ``graph_id``.
That makes an instance safe to embed in a future dashboard without changing
the application's global active dataset.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import logging
from threading import RLock
from typing import Any
from uuid import uuid4

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from dataset_registry import dataset_options, get_record, input_payload
from graph_workspace import _plotly_recovery_script
from utils import _make_error_notif, read_df_from_store

try:
    from multi_axis_engine import build_multi_axis_figure
except ImportError:  # pragma: no cover - allows isolated UI imports during development
    build_multi_axis_figure = None


logger = logging.getLogger(__name__)


# Rebuilding colors, labels or one axis must not repeatedly decode a large
# split-JSON dataframe.  The cache is shared by workspace instances (useful on
# dashboards) and deliberately bounded to the base/filtered pair of datasets.
_FRAME_CACHE_MAX_ENTRIES = 2
_FRAME_CACHE: OrderedDict[str, Any] = OrderedDict()
_FRAME_CACHE_LOCK = RLock()


def _frame_for_payload(payload_ref, payload, meta):
    """Decode one server-side payload once per immutable registry reference."""
    reference = str(payload_ref or "")
    if not reference:
        return read_df_from_store(payload, meta)
    with _FRAME_CACHE_LOCK:
        cached = _FRAME_CACHE.get(reference)
        if cached is not None:
            _FRAME_CACHE.move_to_end(reference)
            return cached
        frame = read_df_from_store(payload, meta)
        _FRAME_CACHE[reference] = frame
        _FRAME_CACHE.move_to_end(reference)
        while len(_FRAME_CACHE) > _FRAME_CACHE_MAX_ENTRIES:
            _FRAME_CACHE.popitem(last=False)
        return frame


TRACE_TYPES = [
    {"label": "Линия", "value": "line"},
    {"label": "Линия + маркеры", "value": "line+markers"},
    {"label": "Маркеры", "value": "scatter"},
    {"label": "Ступенчатая", "value": "step"},
    {"label": "Область", "value": "area"},
    {"label": "Box Plot", "value": "box"},
]

BOX_POINT_OPTIONS = [
    {"label": "Только выбросы", "value": "outliers"},
    {"label": "Все точки", "value": "all"},
    {"label": "Не показывать", "value": "none"},
]

THEMES = [
    {"label": "Plotly", "value": "plotly"},
    {"label": "Plotly White", "value": "plotly_white"},
    {"label": "Seaborn", "value": "seaborn"},
    {"label": "GGPlot", "value": "ggplot2"},
    {"label": "Simple White", "value": "simple_white"},
]

SERIES_COLORS = (
    "#228be6", "#fa5252", "#40c057", "#fd7e14", "#7950f2",
    "#15aabf", "#e64980", "#82c91e", "#fab005", "#4c6ef5",
)


def empty_multi_axis_state(
    *,
    dataset_id=None,
    scope="filtered",
    data_ref=None,
) -> dict[str, Any]:
    """Return the canonical, JSON-safe initial workspace state."""
    return {
        "dataset_id": dataset_id,
        "scope": scope or "filtered",
        # ``dataset_id`` alone cannot identify a freshly reloaded source: the
        # registry intentionally reuses the stable ID ``source``.  Keeping the
        # immutable base reference lets the instance discard mappings that
        # belonged to the previous file without reacting to filter updates.
        "data_ref": data_ref,
        "shared_x": None,
        "show_legend": True,
        "series": [],
        "axes": [],
    }


def _sync_state_for_input(current, dataset_id, scope, data_context):
    """Align instance state with its selected immutable dataset revision.

    The filtered payload reference is deliberately ignored here: changing a
    filter must redraw the same configured pairs, while replacing the base
    dataset must start with a clean graph even though its public ID may still
    be ``source``.
    """
    state = deepcopy(current or empty_multi_axis_state())
    requested_scope = scope or "filtered"
    context = data_context or {}
    context_dataset = context.get("dataset_id")
    context_matches = (
        dataset_id is not None
        and context_dataset is not None
        and str(dataset_id) == str(context_dataset)
    )
    requested_data_ref = context.get("data_ref") if context_matches else None
    previous_dataset = state.get("dataset_id")
    previous_data_ref = state.get("data_ref")

    dataset_changed = (
        previous_dataset is not None
        and dataset_id is not None
        and str(previous_dataset) != str(dataset_id)
    )
    source_replaced = (
        previous_data_ref
        and requested_data_ref
        and str(previous_data_ref) != str(requested_data_ref)
    )
    if dataset_changed or source_replaced:
        return empty_multi_axis_state(
            dataset_id=dataset_id,
            scope=requested_scope,
            data_ref=requested_data_ref,
        )

    state["dataset_id"] = dataset_id
    state["scope"] = requested_scope
    if requested_data_ref is not None:
        state["data_ref"] = requested_data_ref
    else:
        state.setdefault("data_ref", None)
    # Older in-memory states may still contain the former per-series X mode.
    # Multi-Y now has one shared X channel by definition.
    for series in state.get("series") or []:
        if isinstance(series, dict):
            series.pop("x", None)
            series.pop("x_mode", None)
            legacy_line_shape = str(series.pop("line_shape", "") or "").lower()
            if "smooth" not in series and legacy_line_shape == "spline":
                series["smooth"] = True
            if (
                str(series.get("type") or "line").lower() == "step"
                and str(series.get("step_shape") or "").lower() == "spline"
            ):
                series["step_shape"] = "hv"
    for axis in state.get("axes") or []:
        if not isinstance(axis, dict):
            continue
        # Manual bounds and independent axis visibility were removed from the
        # workspace UI. Plotly zoom/reset now owns the visible range, while
        # the series switch controls the pair as one unit.
        for legacy_key in (
            "title", "range", "min", "max", "autorange", "range_auto", "visible"
        ):
            axis.pop(legacy_key, None)
    return state


def _empty_figure(message="Перетащите числовой канал в левую или правую Y-зону"):
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": "#868e96"},
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 78, "r": 78, "t": 48, "b": 72},
        xaxis={"visible": False},
        yaxis={"visible": False},
        uirevision="multi-axis-empty",
    )
    return figure


class MultiYAxisWorkspace:
    """Self-contained multi-Y graph suitable for standalone pages/dashboards."""

    def __init__(
        self,
        *,
        graph_id: str = "multi-y-graph",
        dataset_registry_id: str = "dataset-registry",
        active_dataset_id: str = "active-dataset-id",
        notifications_id: str = "notifications-container",
        location_id: str = "url",
        route_path: str = "/multi-y",
        active_filtered_data_id: str = "filtered-data",
        initial_height: int = 750,
        initial_width: int | None = None,
    ):
        if not graph_id:
            raise ValueError("MultiYAxisWorkspace needs a non-empty graph_id")

        self.graph_id = graph_id
        self.dataset_registry_id = dataset_registry_id
        self.active_dataset_id = active_dataset_id
        self.notifications_id = notifications_id
        self.location_id = location_id
        self.route_path = route_path
        self.active_filtered_data_id = active_filtered_data_id
        self.initial_height = int(initial_height)
        self.initial_width = initial_width
        self.component_id = f"{graph_id}-component"
        self.paper_id = f"{graph_id}-paper"
        self.workspace_id = f"{graph_id}-workspace"
        self._callbacks_registered = False

        self.ids = {
            "state": f"{graph_id}-state",
            "data_context": f"{graph_id}-data-context",
            "sync": f"{graph_id}-sync",
            "revision": f"{graph_id}-revision",
            "dataset": f"{graph_id}-dataset",
            "scope": f"{graph_id}-scope",
            "theme": f"{graph_id}-theme",
            "render_mode": f"{graph_id}-render-mode",
            "height": f"{graph_id}-height",
            "width": f"{graph_id}-width",
            "settings": f"{graph_id}-settings",
            "settings_close_outside": f"{graph_id}-settings-close-outside",
            "series_cards": f"{graph_id}-series-cards",
            "series_chips": f"{graph_id}-series-chips",
            "shared_x": f"{graph_id}-shared-x",
            "show_legend": f"{graph_id}-show-legend",
            "add_y": f"{graph_id}-add-y",
            "add_side": f"{graph_id}-add-side",
            "add_series": f"{graph_id}-add-series",
            "update": f"{graph_id}-update",
            "copy_png": f"{graph_id}-copy-png",
            "download_html": f"{graph_id}-download-html",
            "download": f"{graph_id}-download",
            "clear": f"{graph_id}-clear",
            "open_settings": f"{graph_id}-open-settings",
        }

        self.pattern_types = {
            key: f"{graph_id}-series-{key}"
            for key in (
                "y",
                "trace-type",
                "name",
                "color",
                "smooth",
                "box-points",
                "side",
                "visible",
                "axis-scale",
                "delete",
            )
        }

    @staticmethod
    def _size(value, fallback):
        return f"{value}px" if value is not None else fallback

    def pattern_id(self, key: str, index=ALL):
        """Return a namespaced Dash pattern ID for one series control."""
        return {"type": self.pattern_types[key], "index": index}

    def _drop_zone(self, kind: str, label: str, *, side=None):
        attributes = {
            "data-multi-axis-drop": kind,
            "data-state-store-id": self.ids["state"],
            "data-dataset-control-id": self.ids["dataset"],
            "data-default-label": label,
        }
        if side:
            attributes["data-axis-side"] = side
        return html.Div(
            [
                html.Span(label, className="multi-axis-drop-label"),
                html.Span(
                    "Общий канал" if kind == "x" else "Перетащите Y",
                    className="multi-axis-drop-value",
                ),
                html.Button(
                    "×",
                    type="button",
                    className="multi-axis-drop-clear",
                    title="Очистить общий X",
                    **{"aria-label": "Очистить общий X"},
                ) if kind == "x" else None,
            ],
            className=f"multi-axis-drop-zone multi-axis-drop-zone--{kind}"
            + (f" multi-axis-drop-zone--{side}" if side else ""),
            **attributes,
        )

    def _settings_panel(self):
        return html.Section(
            id=self.ids["settings"],
            className="graph-settings-popover multi-axis-settings-popover",
            role="dialog",
            **{
                "aria-hidden": "true",
                "aria-modal": "false",
                "data-close-on-outside-id": self.ids["settings_close_outside"],
            },
            children=[
                html.Header(
                    [
                        html.Div(
                            [
                                html.Span("⠿", className="graph-settings-drag-handle"),
                                html.Div(
                                    [
                                        html.Div(
                                            "Общие настройки",
                                            className="graph-settings-heading graph-settings-heading--common",
                                        ),
                                        html.Div(
                                            "Настройки Multi-Y",
                                            className="graph-settings-heading graph-settings-heading--specific",
                                        ),
                                        html.Small(
                                            "Настройки этого экземпляра графика",
                                            className="multi-axis-settings-subtitle",
                                        ),
                                    ]
                                ),
                            ],
                            className="graph-settings-heading-wrap",
                        ),
                        html.Button(
                            "×",
                            type="button",
                            className="graph-settings-popover-close",
                            title="Закрыть",
                            **{"aria-label": "Закрыть настройки"},
                        ),
                    ],
                    className="graph-settings-popover-header",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Все серии используют общий X; стиль и Y-ось настраиваются отдельно.",
                                    className="multi-axis-manager-intro",
                                ),
                                dmc.Switch(
                                    id=self.ids["show_legend"],
                                    label="Легенда",
                                    checked=True,
                                    size="xs",
                                    className="multi-axis-legend-switch",
                                ),
                            ],
                            className="multi-axis-manager-toolbar",
                        ),
                        html.Details(
                            [
                                html.Summary("Общие настройки"),
                                html.Div(
                                    self._common_settings(),
                                    className="multi-axis-common-details-body",
                                ),
                            ],
                            className="multi-axis-common-details",
                        ),
                        html.Div(
                            [
                                dmc.Select(
                                    id=self.ids["shared_x"],
                                    label="Общий X",
                                    data=[],
                                    value=None,
                                    searchable=True,
                                    clearable=True,
                                    size="xs",
                                    comboboxProps={"shadow": "md", "withinPortal": False, "zIndex": 10060},
                                ),
                                dmc.Select(
                                    id=self.ids["add_y"],
                                    label="Новая Y-серия",
                                    data=[],
                                    value=None,
                                    searchable=True,
                                    clearable=True,
                                    size="xs",
                                    comboboxProps={"shadow": "md", "withinPortal": False, "zIndex": 10060},
                                ),
                                dmc.SegmentedControl(
                                    id=self.ids["add_side"],
                                    value="left",
                                    data=[
                                        {"label": "Слева", "value": "left"},
                                        {"label": "Справа", "value": "right"},
                                    ],
                                    size="xs",
                                    fullWidth=True,
                                ),
                                dmc.Button(
                                    "Добавить",
                                    id=self.ids["add_series"],
                                    n_clicks=0,
                                    size="xs",
                                    variant="light",
                                    fullWidth=True,
                                ),
                            ],
                            className="multi-axis-add-controls",
                        ),
                        html.Div(id=self.ids["series_cards"], className="multi-axis-series-cards"),
                    ],
                    className="graph-settings-specific graph-settings-scroll multi-axis-series-manager",
                ),
                html.Footer(
                    [
                        html.Span("Изменения применяются автоматически"),
                        dmc.Checkbox(
                            id=self.ids["settings_close_outside"],
                            label="Закрывать кликом вне",
                            checked=False,
                            size="xs",
                        ),
                    ],
                    className="graph-settings-footer multi-axis-settings-footer",
                ),
            ],
        )

    def _common_settings(self):
        return [
            html.Div(
                [
                    dmc.Select(
                        id=self.ids["dataset"],
                        label="Датасет",
                        data=[],
                        value=None,
                        searchable=True,
                        allowDeselect=False,
                        size="xs",
                        comboboxProps={"shadow": "md", "withinPortal": False, "zIndex": 10050},
                    ),
                    dmc.SegmentedControl(
                        id=self.ids["scope"],
                        value="filtered",
                        data=[
                            {"label": "Исходный", "value": "base"},
                            {"label": "Фильтрованный", "value": "filtered"},
                        ],
                        size="xs",
                        fullWidth=True,
                        className="multi-axis-scope-control",
                    ),
                ],
                className="multi-axis-settings-grid multi-axis-settings-grid--dataset",
            ),
            html.Div(
                [
                    dmc.Select(
                        id=self.ids["theme"],
                        label="Тема",
                        data=THEMES,
                        value="plotly",
                        allowDeselect=False,
                        size="xs",
                        comboboxProps={"shadow": "md", "withinPortal": False, "zIndex": 10050},
                    ),
                    dmc.SegmentedControl(
                        id=self.ids["render_mode"],
                        value="hybrid",
                        data=[
                            {"label": "Гибрид", "value": "hybrid"},
                            {"label": "SVG", "value": "svg"},
                        ],
                        size="xs",
                        fullWidth=True,
                        className="multi-axis-render-control",
                    ),
                ],
                className="multi-axis-settings-grid",
            ),
            html.Div(
                [
                    dmc.NumberInput(
                        id=self.ids["height"],
                        label="Высота, px",
                        value=self.initial_height,
                        min=320,
                        max=2400,
                        step=25,
                        size="xs",
                    ),
                    dmc.NumberInput(
                        id=self.ids["width"],
                        label="Ширина, px",
                        value=self.initial_width,
                        min=420,
                        max=4000,
                        step=25,
                        placeholder="Авто",
                        size="xs",
                    ),
                ],
                className="multi-axis-settings-grid",
            ),
        ]

    def _series_card(self, series, axis, numeric_columns):
        series_id = str(series.get("id"))
        color = series.get("color") or "#228be6"
        visible = series.get("visible", True) is not False

        return html.Article(
            [
                html.Header(
                    [
                        html.Span(
                            className="multi-axis-card-color",
                            style={"backgroundColor": color},
                            **{"aria-hidden": "true"},
                        ),
                        html.Div(
                            [
                                html.Strong(series.get("name") or series.get("y") or "Серия"),
                                html.Small(
                                    f"{series.get('y') or 'Y не выбран'} · "
                                    f"{'слева' if axis.get('side') != 'right' else 'справа'}"
                                ),
                            ],
                            className="multi-axis-card-heading",
                        ),
                        dmc.SegmentedControl(
                            id=self.pattern_id("side", series_id),
                            value=axis.get("side") or series.get("side") or "left",
                            data=[
                                {"label": "Слева", "value": "left"},
                                {"label": "Справа", "value": "right"},
                            ],
                            size="xs",
                            className="multi-axis-card-side",
                        ),
                        dmc.Switch(
                            id=self.pattern_id("visible", series_id),
                            checked=visible,
                            size="xs",
                            **{"aria-label": "Показывать серию"},
                        ),
                        html.Button(
                            "×",
                            id=self.pattern_id("delete", series_id),
                            type="button",
                            n_clicks=0,
                            className="multi-axis-series-delete",
                            title="Удалить серию и её ось",
                            **{"aria-label": "Удалить серию и её ось"},
                        ),
                    ],
                    className="multi-axis-card-header",
                ),
                html.Div(
                    [
                        dmc.Select(
                            id=self.pattern_id("y", series_id),
                            label="Y",
                            data=numeric_columns,
                            value=series.get("y"),
                            searchable=True,
                            allowDeselect=False,
                            size="xs",
                            comboboxProps={"shadow": "md", "withinPortal": False, "zIndex": 10060},
                        ),
                        dmc.TextInput(
                            id=self.pattern_id("name", series_id),
                            label="Подпись оси",
                            value=series.get("name") or series.get("y") or "",
                            size="xs",
                        ),
                        dmc.Select(
                            id=self.pattern_id("trace-type", series_id),
                            label="Отображение",
                            data=TRACE_TYPES,
                            value=series.get("type") or "line",
                            allowDeselect=False,
                            size="xs",
                            comboboxProps={"shadow": "md", "withinPortal": False, "zIndex": 10060},
                        ),
                        dmc.ColorInput(
                            id=self.pattern_id("color", series_id),
                            label="Цвет",
                            value=color,
                            format="hex",
                            size="xs",
                            popoverProps={"withinPortal": False, "zIndex": 10060},
                        ),
                    ],
                    className="multi-axis-card-grid",
                ),
                html.Details(
                    [
                        html.Summary("Шкала и линия"),
                        html.Div(
                            [
                                dmc.SegmentedControl(
                                    id=self.pattern_id("axis-scale", series_id),
                                    value=axis.get("type") or "linear",
                                    data=[
                                        {"label": "Линейная", "value": "linear"},
                                        {"label": "Логарифм.", "value": "log"},
                                    ],
                                    size="xs",
                                    fullWidth=True,
                                    className="multi-axis-scale-control",
                                ),
                                dmc.Switch(
                                    id=self.pattern_id("smooth", series_id),
                                    label="Сглаживание",
                                    checked=bool(series.get("smooth", False)),
                                    disabled=(series.get("type") or "line") not in {
                                        "line", "line+markers", "area"
                                    },
                                    size="xs",
                                    className="multi-axis-smooth-switch",
                                ),
                                dmc.Select(
                                    id=self.pattern_id("box-points", series_id),
                                    label="Точки Box Plot",
                                    data=BOX_POINT_OPTIONS,
                                    value=(
                                        "none"
                                        if series.get("box_points") is False
                                        else series.get("box_points") or "outliers"
                                    ),
                                    allowDeselect=False,
                                    size="xs",
                                    className="multi-axis-box-points",
                                    style={
                                        "display": "block"
                                        if (series.get("type") or "line") == "box"
                                        else "none"
                                    },
                                    comboboxProps={
                                        "shadow": "md",
                                        "withinPortal": False,
                                        "zIndex": 10060,
                                    },
                                ),
                            ],
                            className="multi-axis-scale-options",
                        ),
                    ],
                    className="multi-axis-axis-details",
                ),
            ],
            id={"type": f"{self.graph_id}-series-card", "index": series_id},
            className="multi-axis-series-card",
            **{"data-series-id": series_id},
        )

    def _render_paper(self):
        workspace_data = {
            "data-graph-id": self.graph_id,
            "data-columns-container-id": "columns-badges",
            "data-multi-axis-state-id": self.ids["state"],
            "data-multi-axis-dataset-id": self.ids["dataset"],
            "data-settings-popup-id": self.ids["settings"],
            "data-settings-button-title": "Настройки Multi-Y",
            "data-common-settings-in-specific": "true",
            "data-action-open-specific-settings": self.ids["open_settings"],
            "data-action-refresh": self.ids["update"],
            "data-action-download-html": self.ids["download_html"],
            "data-action-copy-png": self.ids["copy_png"],
            "data-action-clear-graph": self.ids["clear"],
        }

        workspace = html.Div(
            id=self.workspace_id,
            className="graph-workspace multi-axis-workspace",
            **workspace_data,
            children=[
                dcc.Loading(
                    dcc.Graph(
                        id=self.graph_id,
                        figure=_empty_figure(),
                        className="graph-workspace-plot multi-axis-plot",
                        config={
                            "displaylogo": False,
                            "displayModeBar": "hover",
                            "scrollZoom": True,
                            "responsive": True,
                            "modeBarButtonsToRemove": [
                                "zoom2d",
                                "pan2d",
                                "zoomIn2d",
                                "zoomOut2d",
                                "autoScale2d",
                                "select2d",
                                "lasso2d",
                            ],
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "multi-y-graph",
                                "scale": 2,
                            },
                        },
                        style={"height": "100%", "width": "100%"},
                    ),
                    type="default",
                    className="graph-workspace-spinner",
                    parent_className="graph-workspace-loading",
                    parent_style={"height": "100%", "width": "100%"},
                ),
                html.Div(
                    [
                        self._drop_zone("y", "Добавить Y", side="left"),
                        self._drop_zone("y", "Добавить Y", side="right"),
                        self._drop_zone("x", "Общий X"),
                    ],
                    className="multi-axis-drop-layer",
                ),
                html.Div(id=self.ids["series_chips"], className="multi-axis-series-chips"),
            ],
        )

        return dmc.Paper(
            id=self.paper_id,
            className="graph-workspace-shell graph-fullscreen-host multi-axis-workspace-shell",
            shadow="md",
            withBorder=True,
            style={
                "height": self._size(self.initial_height, "750px"),
                "width": self._size(self.initial_width, "100%"),
            },
            children=[workspace, self._settings_panel()],
        )

    def _service_components(self):
        return html.Div(
            [
                dcc.Store(id=self.ids["state"], data=empty_multi_axis_state()),
                # Only references and the selected layer live here.  The
                # dataframe itself remains in dataset_registry's server cache.
                dcc.Store(id=self.ids["data_context"]),
                dcc.Store(id=self.ids["revision"], data=0),
                dcc.Store(id=self.ids["sync"]),
                html.Button("update", id=self.ids["update"]),
                html.Button("copy-png", id=self.ids["copy_png"]),
                html.Button("download-html", id=self.ids["download_html"]),
                html.Button("clear", id=self.ids["clear"]),
                html.Button("open-settings", id=self.ids["open_settings"]),
                dcc.Download(id=self.ids["download"]),
            ],
            className="graph-workspace-services",
            style={"display": "none"},
            **{"aria-hidden": "true"},
        )

    def render(self):
        """Return the complete component tree for one workspace instance."""
        return html.Div(
            [self._render_paper(), self._service_components()],
            id=self.component_id,
            className="graph-workspace-component multi-axis-workspace-component",
        )

    def register_callbacks(self, app):
        """Register all callbacks owned by this instance exactly once."""
        if self._callbacks_registered:
            return
        self._callbacks_registered = True

        self._register_dataset_callbacks(app)
        self._register_state_callbacks(app)
        self._register_render_callbacks(app)
        self._register_action_callbacks(app)

    def _register_dataset_callbacks(self, app):
        @app.callback(
            Output(self.ids["dataset"], "data"),
            Output(self.ids["dataset"], "value"),
            Input(self.location_id, "pathname"),
            Input(self.dataset_registry_id, "data"),
            State(self.active_dataset_id, "data"),
            State(self.ids["dataset"], "value"),
        )
        def sync_dataset_selector(pathname, registry, active_id, current):
            if pathname != self.route_path:
                return no_update, no_update
            options = dataset_options(registry)
            available = {item["value"] for item in options}
            selected = current if current in available else active_id
            if selected not in available:
                selected = next(iter(available), None)
            # A Multi-Y instance owns its dataset choice.  Changing the
            # application's global active dataset must not silently retarget
            # an already configured dashboard widget.
            return options, no_update if selected == current else selected

        @app.callback(
            Output(self.ids["data_context"], "data"),
            Input(self.location_id, "pathname"),
            Input(self.ids["dataset"], "value"),
            Input(self.ids["scope"], "value"),
            Input(self.dataset_registry_id, "data"),
            State(self.ids["data_context"], "data"),
        )
        def sync_data_context(pathname, dataset_id, scope, registry, current):
            """Publish a compact rebuild token without copying dataframe JSON."""
            if pathname != self.route_path:
                raise PreventUpdate
            selected_scope = "filtered" if scope == "filtered" else "base"
            record = get_record(registry, dataset_id) or {}
            reference = (
                (record.get("filtered_ref") or record.get("data_ref"))
                if selected_scope == "filtered"
                else record.get("data_ref")
            )
            context = {
                "dataset_id": str(dataset_id) if dataset_id is not None else None,
                "scope": selected_scope,
                "data_ref": record.get("data_ref"),
                "payload_ref": reference,
            }
            return no_update if context == (current or {}) else context

        @app.callback(
            Output(self.ids["state"], "data", allow_duplicate=True),
            Input(self.location_id, "pathname"),
            Input(self.ids["dataset"], "value"),
            Input(self.ids["scope"], "value"),
            Input(self.ids["data_context"], "data"),
            State(self.ids["state"], "data"),
            prevent_initial_call=True,
        )
        def sync_input_state(pathname, dataset_id, scope, data_context, current):
            if pathname != self.route_path:
                raise PreventUpdate
            state = _sync_state_for_input(current, dataset_id, scope, data_context)
            return state if state != (current or empty_multi_axis_state()) else no_update

    def _register_state_callbacks(self, app):
        input_keys = list(self.pattern_types)
        dependencies = [Input(self.pattern_id(key), "n_clicks" if key == "delete" else (
            "checked" if key in {"visible", "smooth"} else "value"
        )) for key in input_keys]

        @app.callback(
            Output(self.ids["state"], "data", allow_duplicate=True),
            *dependencies,
            State(self.ids["state"], "data"),
            prevent_initial_call=True,
        )
        def update_series_from_controls(*values_and_state):
            current = values_and_state[-1] if values_and_state else None
            trigger = ctx.triggered_id
            if not isinstance(trigger, dict):
                raise PreventUpdate

            series_id = str(trigger.get("index") or "")
            trigger_type = trigger.get("type")
            key = next(
                (name for name, pattern_type in self.pattern_types.items() if pattern_type == trigger_type),
                None,
            )
            if not key or not series_id:
                raise PreventUpdate

            value = ctx.triggered[0].get("value") if ctx.triggered else None
            state = deepcopy(current or empty_multi_axis_state())
            series_list = state.setdefault("series", [])
            axes = state.setdefault("axes", [])
            series = next((item for item in series_list if str(item.get("id")) == series_id), None)
            if series is None:
                raise PreventUpdate
            axis_id = str(series.get("axis_id") or f"axis-{series_id}")
            series["axis_id"] = axis_id
            axis = next((item for item in axes if str(item.get("id")) == axis_id), None)
            if axis is None:
                axis = {"id": axis_id, "side": series.get("side") or "left", "type": "linear", "autorange": True}
                axes.append(axis)

            before = deepcopy(state)
            if key == "delete":
                if not value:
                    raise PreventUpdate
                state["series"] = [item for item in series_list if str(item.get("id")) != series_id]
                used_axes = {str(item.get("axis_id")) for item in state["series"]}
                state["axes"] = [item for item in axes if str(item.get("id")) in used_axes]
            elif key == "y":
                old_y = series.get("y")
                old_name = series.get("name")
                series["y"] = value
                if not old_name or old_name == old_y:
                    series["name"] = value
                axis.pop("title", None)
            elif key == "trace-type":
                series["type"] = value or "line"
            elif key == "name":
                series["name"] = value
                axis.pop("title", None)
            elif key == "color":
                series["color"] = value
            elif key == "smooth":
                series["smooth"] = bool(value)
            elif key == "box-points":
                series["box_points"] = False if value == "none" else (value or "outliers")
            elif key == "side":
                side = value or "left"
                series["side"] = side
                axis["side"] = side
            elif key == "visible":
                series["visible"] = bool(value)
            elif key == "axis-scale":
                axis["type"] = value or "linear"

            return state if state != before else no_update

        @app.callback(
            Output(self.ids["state"], "data", allow_duplicate=True),
            Input(self.ids["shared_x"], "value"),
            Input(self.ids["show_legend"], "checked"),
            State(self.location_id, "pathname"),
            State(self.ids["state"], "data"),
            prevent_initial_call=True,
        )
        def update_shared_x_and_legend(value, show_legend, pathname, current):
            if pathname != self.route_path:
                raise PreventUpdate
            state = deepcopy(current or empty_multi_axis_state())
            normalized_value = value if value not in (None, "") else None
            normalized_legend = show_legend is not False
            if (
                state.get("shared_x") == normalized_value
                and state.get("show_legend", True) == normalized_legend
            ):
                raise PreventUpdate
            state["shared_x"] = normalized_value
            state["show_legend"] = normalized_legend
            return state

        @app.callback(
            Output(self.ids["state"], "data", allow_duplicate=True),
            Output(self.ids["add_y"], "value"),
            Input(self.ids["add_series"], "n_clicks"),
            State(self.location_id, "pathname"),
            State(self.ids["add_y"], "value"),
            State(self.ids["add_side"], "value"),
            State(self.ids["dataset"], "value"),
            State(self.ids["scope"], "value"),
            State(self.ids["state"], "data"),
            prevent_initial_call=True,
        )
        def add_series_from_picker(
            n_clicks,
            pathname,
            y_column,
            side,
            dataset_id,
            scope,
            current,
        ):
            if not n_clicks or pathname != self.route_path or not y_column:
                raise PreventUpdate
            state = deepcopy(current or empty_multi_axis_state())
            if state.get("dataset_id") not in (None, dataset_id):
                state = empty_multi_axis_state(dataset_id=dataset_id, scope=scope)
            state["dataset_id"] = dataset_id
            state["scope"] = scope or "filtered"
            series_id = f"series-{uuid4().hex[:12]}"
            axis_id = f"axis-{series_id}"
            selected_side = "right" if side == "right" else "left"
            color = SERIES_COLORS[len(state.get("series") or []) % len(SERIES_COLORS)]
            state.setdefault("series", []).append({
                "id": series_id,
                "y": y_column,
                "type": "scatter",
                "name": y_column,
                "color": color,
                "smooth": False,
                "box_points": "outliers",
                "side": selected_side,
                "axis_id": axis_id,
                "visible": True,
            })
            state.setdefault("axes", []).append({
                "id": axis_id,
                "side": selected_side,
                "type": "linear",
            })
            return state, None

        app.clientside_callback(
            f"""
            function(state, datasetId, scope) {{
                var workspace = document.getElementById({self.workspace_id!r});
                if (!workspace) return window.dash_clientside.no_update;
                var normalized = state || {{shared_x: null, show_legend: true, series: [], axes: []}};
                workspace.setAttribute('data-multi-axis-state', JSON.stringify(normalized));
                workspace.setAttribute('data-selected-dataset', datasetId || '');
                workspace.setAttribute('data-selected-scope', scope || 'filtered');
                var xZone = workspace.querySelector('[data-multi-axis-drop="x"]');
                if (xZone) {{
                    var x = normalized.shared_x || null;
                    xZone.classList.toggle('has-value', Boolean(x));
                    xZone.setAttribute('data-current-value', JSON.stringify(x));
                    var label = xZone.querySelector('.multi-axis-drop-value');
                    if (label) {{ label.textContent = x || 'Общий канал'; label.title = x || ''; }}
                }}
                return Date.now();
            }}
            """,
            Output(self.ids["sync"], "data"),
            Input(self.ids["state"], "data"),
            Input(self.ids["dataset"], "value"),
            Input(self.ids["scope"], "value"),
        )

    def _register_render_callbacks(self, app):
        @app.callback(
            Output(self.ids["series_cards"], "children"),
            Output(self.ids["series_chips"], "children"),
            Output(self.ids["shared_x"], "data"),
            Output(self.ids["shared_x"], "value"),
            Output(self.ids["add_y"], "data"),
            Output(self.ids["show_legend"], "checked"),
            Input(self.location_id, "pathname"),
            Input(self.ids["state"], "data"),
            Input(self.ids["dataset"], "value"),
            Input(self.dataset_registry_id, "data"),
        )
        def render_series_manager(pathname, state, dataset_id, registry):
            if pathname != self.route_path:
                return (no_update,) * 6
            state = state or empty_multi_axis_state()
            series_list = state.get("series") or []
            record = get_record(registry, dataset_id) or {}
            meta = record.get("meta") or {}
            raw_columns = [str(value) for value in (meta.get("columns") or [])]
            raw_numeric = [str(value) for value in (meta.get("numeric") or [])]
            columns = [{"label": value, "value": value} for value in raw_columns]
            numeric = [{"label": value, "value": value} for value in raw_numeric]
            axes_by_id = {str(item.get("id")): item for item in (state.get("axes") or [])}

            if not series_list:
                cards = html.Div(
                    [
                        html.Strong("Серий пока нет"),
                        html.Span("Перетащите числовой канал на левую или правую Y-зону графика."),
                    ],
                    className="multi-axis-series-empty",
                )
                return cards, [], columns, state.get("shared_x"), numeric, state.get("show_legend", True)

            cards = []
            chips = []
            for item in series_list:
                series = dict(item or {})
                axis_id = str(series.get("axis_id") or f"axis-{series.get('id')}")
                axis = dict(axes_by_id.get(axis_id) or {
                    "id": axis_id,
                    "side": series.get("side") or "left",
                    "type": "linear",
                })
                cards.append(self._series_card(series, axis, numeric))
                color = series.get("color") or "#228be6"
                chips.append(
                    html.Button(
                        [
                            html.Span(className="multi-axis-chip-dot", style={"backgroundColor": color}),
                            html.Span(series.get("name") or series.get("y") or "Серия"),
                            html.Small("L" if axis.get("side") != "right" else "R"),
                        ],
                        type="button",
                        className="multi-axis-series-chip"
                        + (" is-hidden" if series.get("visible", True) is False else ""),
                        title="Открыть настройки серии",
                        style={"--series-color": color},
                        **{"data-series-id": str(series.get("id"))},
                    )
                )
            return cards, chips, columns, state.get("shared_x"), numeric, state.get("show_legend", True)

        @app.callback(
            Output(self.graph_id, "figure"),
            Output(self.notifications_id, "sendNotifications", allow_duplicate=True),
            Input(self.location_id, "pathname"),
            Input(self.ids["state"], "data"),
            Input(self.ids["data_context"], "data"),
            Input(self.ids["theme"], "value"),
            Input(self.ids["render_mode"], "value"),
            Input(self.ids["height"], "value"),
            Input(self.ids["width"], "value"),
            Input(self.ids["revision"], "data"),
            State(self.dataset_registry_id, "data"),
            prevent_initial_call="initial_duplicate",
        )
        def build_figure(
            pathname,
            state,
            data_context,
            theme,
            render_mode,
            height,
            width,
            _revision,
            registry,
        ):
            if pathname != self.route_path:
                raise PreventUpdate
            data_context = data_context or {}
            dataset_id = data_context.get("dataset_id")
            scope = data_context.get("scope") or "filtered"
            state_dataset = (state or {}).get("dataset_id")
            if state_dataset and str(state_dataset) != str(dataset_id):
                # DnD can update the instance state and its dataset selector in
                # the same browser event.  Wait for the compact context token
                # instead of rendering one transient mixed-dataset figure.
                raise PreventUpdate
            state_data_ref = (state or {}).get("data_ref")
            context_data_ref = data_context.get("data_ref")
            if (
                state_data_ref
                and context_data_ref
                and str(state_data_ref) != str(context_data_ref)
            ):
                # A new source file can keep the public dataset ID ``source``.
                # Wait for sync_input_state to clear the old channel mapping.
                raise PreventUpdate
            series = (state or {}).get("series") or []
            if not series:
                figure = _empty_figure()
                figure.update_layout(template=theme or "plotly")
                return figure, []

            record = get_record(registry, dataset_id)
            if not record:
                return _empty_figure("Выберите доступный датасет"), []
            payload, meta = input_payload(
                registry,
                dataset_id,
                scope or "filtered",
            )
            if not payload:
                return _empty_figure("В выбранном слое датасета нет данных"), []
            if build_multi_axis_figure is None:
                return _empty_figure("Модуль многоосевого графика не загружен"), _make_error_notif(
                    "Модуль multi_axis_engine недоступен."
                )

            try:
                frame = _frame_for_payload(
                    data_context.get("payload_ref"),
                    payload,
                    meta or record.get("meta") or {},
                )
                render_state = deepcopy(state or empty_multi_axis_state())
                render_state["height"] = height or self.initial_height
                render_state["width"] = width
                result = build_multi_axis_figure(
                    frame,
                    render_state,
                    template=theme or "plotly",
                    render_mode=render_mode or "hybrid",
                )
                figure = result.figure if hasattr(result, "figure") else result
                if isinstance(figure, go.Figure):
                    figure.update_layout(template=theme or "plotly")
                sample_message = (
                    getattr(result, "metadata", {}) or {}
                ).get("visual_sample_message")
                notifications = []
                if sample_message:
                    notifications.append({
                        "id": f"{self.graph_id}-visual-sample",
                        "title": "Multi-Y: визуальная выборка",
                        "message": sample_message,
                        "color": "blue",
                        "action": "show",
                        "position": "bottom-right",
                        "autoClose": False,
                        "withCloseButton": True,
                    })
                return figure, notifications
            except Exception as error:
                logger.error("Multi-Y figure build failed: %s", error, exc_info=True)
                error_figure = _empty_figure("Не удалось построить Multi-Y")
                error_figure.update_layout(template=theme or "plotly")
                return error_figure, _make_error_notif(
                    f"Не удалось построить многоосевой график: {error}"
                )

    def _register_action_callbacks(self, app):
        recovery_script = _plotly_recovery_script(self.graph_id)
        clear_script = """
            function(nClicks, state, datasetId, scope) {
                if (!nClicks) return window.dash_clientside.no_update;
        """ + recovery_script + """
                return {
                    dataset_id: datasetId || (state && state.dataset_id) || null,
                    scope: scope || (state && state.scope) || 'filtered',
                    data_ref: (state && state.data_ref) || null,
                    shared_x: null,
                    show_legend: true,
                    series: [],
                    axes: []
                };
            }
        """

        app.clientside_callback(
            """
            function(nClicks, current) {
                if (!nClicks) return window.dash_clientside.no_update;
                return (Number(current) || 0) + 1;
            }
            """,
            Output(self.ids["revision"], "data"),
            Input(self.ids["update"], "n_clicks"),
            State(self.ids["revision"], "data"),
            prevent_initial_call=True,
        )

        app.clientside_callback(
            clear_script,
            Output(self.ids["state"], "data", allow_duplicate=True),
            Input(self.ids["clear"], "n_clicks"),
            State(self.ids["state"], "data"),
            State(self.ids["dataset"], "value"),
            State(self.ids["scope"], "value"),
            prevent_initial_call=True,
        )

        app.clientside_callback(
            f"""
            function(nClicks, figure) {{
                if (!nClicks || !figure) throw window.dash_clientside.PreventUpdate;
                function notice(title, message, color) {{
                    return [{{
                        id: crypto.randomUUID(), title: title, message: message,
                        color: color, action: 'show', autoClose: 4500
                    }}];
                }}
                if (!window.graphPng) {{
                    return notice('PNG не скопирован', 'Модуль экспорта не загружен.', 'red');
                }}
                return window.graphPng.copyToClipboard({self.graph_id!r}).then(
                    function() {{ return notice('PNG скопирован', 'График помещён в буфер обмена.', 'green'); }},
                    function(error) {{ return notice('PNG не скопирован', error.message || 'Ошибка экспорта.', 'red'); }}
                );
            }}
            """,
            Output(self.notifications_id, "sendNotifications", allow_duplicate=True),
            Input(self.ids["copy_png"], "n_clicks"),
            State(self.graph_id, "figure"),
            prevent_initial_call=True,
        )

        @app.callback(
            Output(self.ids["download"], "data"),
            Output(self.notifications_id, "sendNotifications", allow_duplicate=True),
            Input(self.ids["download_html"], "n_clicks"),
            State(self.graph_id, "figure"),
            prevent_initial_call=True,
        )
        def download_html(n_clicks, figure):
            if not n_clicks or not figure:
                raise PreventUpdate
            try:
                content = go.Figure(figure).to_html(include_plotlyjs="cdn")
                return {
                    "content": content,
                    "filename": f"{self.graph_id}.html",
                    "type": "text/html",
                }, []
            except Exception as error:
                return no_update, _make_error_notif(f"Ошибка экспорта HTML: {error}")

        app.clientside_callback(
            """
            function(height, width) {
                function size(value, fallback) {
                    var number = Number(value);
                    return Number.isFinite(number) && number > 0
                        ? Math.round(number) + 'px'
                        : fallback;
                }
                return {height: size(height, '750px'), width: size(width, '100%')};
            }
            """,
            Output(self.paper_id, "style"),
            Input(self.ids["height"], "value"),
            Input(self.ids["width"], "value"),
        )


# Ready-to-use permanent page instance.  Additional dashboard instances should
# instantiate the class with another graph_id and register their callbacks.
multi_y_workspace = MultiYAxisWorkspace()


__all__ = [
    "MultiYAxisWorkspace",
    "empty_multi_axis_state",
    "multi_y_workspace",
]
