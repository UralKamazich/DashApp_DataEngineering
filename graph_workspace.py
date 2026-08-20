# -*- coding: utf-8 -*-
"""Reusable, self-contained visual workspace for a Plotly graph.

GraphWorkspace is the public integration point. A dashboard supplies data
stores and a figure-building callback, while the component owns presentation,
field assignment, settings, export actions and interaction state.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc

from config import COLOR_THRESHOLD
from graph_help import GRAPH_HELP_OPTIONS, GRAPH_INSTRUCTIONS, render_instruction
from utils import _make_error_notif, apply_custom_colors_safely


logger = logging.getLogger(__name__)


DEFAULT_FIELDS = (
    {"key": "x", "label": "X", "target": "dropdown_x", "zone": "axis-x"},
    {"key": "y", "label": "Y", "target": "dropdown_y", "zone": "axis-y"},
    {"key": "z", "label": "Z", "target": "dropdown_z", "zone": "secondary"},
    {"key": "color", "label": "Цвет", "target": "dropdown_color", "zone": "secondary"},
    {"key": "size", "label": "Размер", "target": "dropdown_size", "zone": "secondary"},
    {"key": "text", "label": "Подпись", "target": "dropdown_text", "zone": "secondary"},
    {"key": "facet-row", "label": "Facet row", "target": "dropdown_facet_row", "zone": "secondary"},
    {"key": "facet-col", "label": "Facet col", "target": "dropdown_facet_col", "zone": "secondary"},
    {
        "key": "hover",
        "label": "Hover",
        "target": "dropdown_hover_data",
        "zone": "secondary",
        "mode": "append",
    },
)


# Backward-compatible IDs for the current application. New dashboards can
# omit this mapping and receive IDs namespaced by graph_id automatically.
LEGACY_GRAPH_ACTION_IDS = {
    "update": "update-graf",
    "copy_png": "copy-png-button",
    "open_settings": "context-menu-btn",
    "download_html": "download-button",
    "download_component": "download-file",
    "save_png": "save-png-button",
    "change_colors": "shuffle-button",
    "clear": "clear-graph-button",
    "view_revision": "graph-view-revision",
    "color_modal": "color-modal",
    "color_inputs": "color-inputs",
    "color_mode": "color-mode-toggle",
    "apply_colors": "apply-colors",
    "custom_colors": "custom-colors",
    "sync": "graph-workspace-sync",
}


def _component_id(component) -> str:
    component_id = getattr(component, "id", None)
    if not isinstance(component_id, str) or not component_id:
        raise ValueError("GraphWorkspace controls must have a string id")
    return component_id


def _plotly_recovery_script(graph_id: str) -> str:
    """Return client-side statements that reset a failed Plotly.react cycle."""
    return """
                var graphRoot = document.getElementById(%s);
                var plot = graphRoot && graphRoot.querySelector('.js-plotly-plot');
                if (plot && window.Plotly && typeof window.Plotly.purge === 'function') {
                    try {
                        window.Plotly.purge(plot);
                    } catch (error) {
                        console.warn('Plotly hard reset failed:', error);
                    }
                }
                if (graphRoot) {
                    graphRoot.classList.remove('dash-graph--pending');
                    graphRoot.querySelectorAll('.dash-graph--pending').forEach(function(element) {
                        element.classList.remove('dash-graph--pending');
                    });
                }
    """ % json.dumps(graph_id)


class GraphWorkspace:
    """Build and register a complete graph workspace.

    Field and chart-type controls are dependencies because their option lists
    normally come from a dashboard's data model. They are still rendered and
    managed inside this component.
    """

    def __init__(
        self,
        *,
        graph_id: str,
        chart_type_control,
        field_controls: Mapping[str, object],
        settings_panel=None,
        fields=DEFAULT_FIELDS,
        action_ids: Mapping[str, str] | None = None,
        columns_container_id: str = "columns-badges",
        notifications_id: str = "notifications-container",
        initial_height: int = 750,
        initial_width: int | None = None,
        include_color_controls: bool | None = None,
    ):
        self.graph_id = graph_id
        self.component_id = f"{graph_id}-component"
        self.paper_id = f"{graph_id}-paper"
        self.workspace_id = f"{graph_id}-workspace"
        self.chart_type_control = chart_type_control
        self.chart_type_id = _component_id(chart_type_control)
        self.field_controls = dict(field_controls)
        self.fields = tuple(fields)
        self._fields_by_key = {}
        for field in self.fields:
            key = field.get("key")
            target = field.get("target")
            if not key or not target:
                raise ValueError("Each graph field needs non-empty 'key' and 'target'")
            if key in self._fields_by_key:
                raise ValueError(f"Duplicate graph field key: {key!r}")
            self._fields_by_key[key] = field
        self.settings_panel = settings_panel
        if self.settings_panel is not None:
            self.settings_panel.bind_namespace(graph_id)
        self.columns_container_id = columns_container_id
        self.notifications_id = notifications_id
        self.initial_height = initial_height
        self.initial_width = initial_width
        self.include_color_controls = (
            self.settings_panel is not None
            if include_color_controls is None
            else bool(include_color_controls)
        )
        if self.include_color_controls and self.settings_panel is None:
            raise ValueError("Color controls require a GraphSettingsPanel")
        self._callbacks_registered = False

        required = {field["target"] for field in self.fields}
        missing = required.difference(self.field_controls)
        if missing:
            raise ValueError(f"Missing graph field controls: {sorted(missing)}")

        self.field_ids = {
            key: _component_id(control)
            for key, control in self.field_controls.items()
        }
        self.field_ids_by_key = {
            key: self.field_ids[field["target"]]
            for key, field in self._fields_by_key.items()
        }
        generated_ids = {
            "update": f"{graph_id}-update",
            "copy_png": f"{graph_id}-copy-png",
            "open_settings": f"{graph_id}-open-settings",
            "download_html": f"{graph_id}-download-html",
            "download_component": f"{graph_id}-download",
            "save_png": f"{graph_id}-save-png",
            "change_colors": f"{graph_id}-change-colors",
            "clear": f"{graph_id}-clear",
            "view_revision": f"{graph_id}-view-revision",
            "color_modal": f"{graph_id}-color-modal",
            "color_inputs": f"{graph_id}-color-inputs",
            "color_mode": f"{graph_id}-color-mode",
            "apply_colors": f"{graph_id}-apply-colors",
            "custom_colors": f"{graph_id}-custom-colors",
            "sync": f"{graph_id}-sync",
            "help": f"{graph_id}-help",
            "help_modal": f"{graph_id}-help-modal",
            "help_type": f"{graph_id}-help-type",
            "help_content": f"{graph_id}-help-content",
            "help_close": f"{graph_id}-help-close",
        }
        generated_ids.update(action_ids or {})
        self.ids = generated_ids

        # Справка показывает только те типы, которые доступны в этом
        # воркспейсе (берутся из опций селекта типа графика).
        help_options = GRAPH_HELP_OPTIONS
        control_data = getattr(self.chart_type_control, "data", None)
        if isinstance(control_data, list) and control_data:
            allowed = {
                item.get("value") for item in control_data if isinstance(item, dict)
            }
            if allowed:
                filtered = [o for o in GRAPH_HELP_OPTIONS if o["value"] in allowed]
                if filtered:
                    help_options = filtered
        self.help_options = help_options
        self.help_default_type = help_options[0]["value"]

    @staticmethod
    def _pixel_size(value: int | None, fallback: str) -> str:
        return f"{value}px" if value is not None else fallback

    def _drop_zone(self, field: Mapping[str, str]):
        target_id = self.field_ids[field["target"]]
        return html.Div(
            [
                html.Span(field["label"], className="graph-drop-zone-name"),
                html.Span("Не выбрано", className="graph-drop-zone-value"),
                html.Button(
                    "×",
                    type="button",
                    className="graph-zone-clear",
                    title=f"Очистить поле {field['label']}",
                    **{"aria-label": f"Очистить поле {field['label']}"},
                ),
            ],
            className=(
                f"graph-drop-zone graph-drop-zone--{field['zone']} "
                f"graph-drop-zone--{field['key']}"
            ),
            **{
                "data-drop-target": target_id,
                "data-drop-mode": field.get("mode", "replace"),
                "data-default-label": field["label"],
            },
        )

    @staticmethod
    def _action_button(component_id: str, symbol: str, label: str):
        return dmc.Tooltip(
            label=label,
            withArrow=True,
            openDelay=300,
            children=dmc.ActionIcon(
                html.Span(symbol, className="graph-toolbar-symbol"),
                id=component_id,
                variant="subtle",
                color="gray",
                size="lg",
                radius="md",
            ),
        )

    def _render_paper(self):
        secondary = [self._drop_zone(field) for field in self.fields if field["zone"] == "secondary"]
        axes = [self._drop_zone(field) for field in self.fields if field["zone"] != "secondary"]

        workspace_data = {
            "data-graph-id": self.graph_id,
            "data-columns-container-id": self.columns_container_id,
            "data-action-refresh": self.ids["update"],
            "data-action-download-html": self.ids["download_html"],
            "data-action-copy-png": self.ids["copy_png"],
            "data-action-save-png": self.ids["save_png"],
            "data-action-clear-graph": self.ids["clear"],
        }
        if self.include_color_controls:
            workspace_data["data-action-change-colors"] = self.ids["change_colors"]
        if self.settings_panel is not None:
            workspace_data.update({
                "data-settings-popup-id": self.settings_id("graph-settings-popover"),
                "data-action-open-specific-settings": self.ids["open_settings"],
            })

        return dmc.Paper(
            id=self.paper_id,
            className="graph-workspace-shell",
            shadow="md",
            withBorder=True,
            style={
                "height": self._pixel_size(self.initial_height, "750px"),
                "width": self._pixel_size(self.initial_width, "100%"),
            },
            children=html.Div(
                id=self.workspace_id,
                className="graph-workspace",
                **workspace_data,
                children=[
                    # Toolbar and drop zones are siblings of dcc.Graph, so PNG
                    # and HTML export contain the Plotly figure only.
                    html.Div(
                        className="graph-workspace-toolbar",
                        children=[
                            html.Div(self.chart_type_control, className="graph-type-control"),
                            html.Div(className="graph-workspace-toolbar-separator"),
                            *(
                                [self._action_button(self.ids["open_settings"], "⚙", "Настройки типа графика")]
                                if self.settings_panel is not None
                                else []
                            ),
                            self._action_button(self.ids["help"], "?", "Инструкция по типу графика"),
                        ],
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            figure={},
                            id=self.graph_id,
                            className="graph-workspace-plot",
                            config={
                                "displaylogo": False,
                                "modeBarButtonsToRemove": [],
                                "modeBarButtonsToAdd": ["fullscreen"],
                                "displayModeBar": True,
                                "scrollZoom": True,
                                "responsive": True,
                            },
                            style={"height": "100%", "width": "100%"},
                        ),
                        type="default",
                        className="graph-workspace-spinner",
                        parent_className="graph-workspace-loading",
                        parent_style={"height": "100%", "width": "100%"},
                    ),
                    html.Div(
                        className="graph-drop-layer",
                        children=[*axes, html.Div(secondary, className="graph-drop-secondary")],
                    ),
                    html.Div(
                        list(self.field_controls.values()),
                        className="graph-field-state",
                        **{"aria-hidden": "true"},
                    ),
                ],
            ),
        )

    def _color_modal(self):
        return dmc.Modal(
            id=self.ids["color_modal"],
            title="Выберите цвета для классов",
            opened=False,
            size="auto",
            children=[
                dmc.Group(
                    [
                        dmc.Text("Режим выбора цвета:"),
                        dmc.Switch(
                            id=self.ids["color_mode"],
                            onLabel="Ручной",
                            offLabel="Авто",
                            checked=False,
                            size="md",
                        ),
                    ]
                ),
                dmc.Stack(id=self.ids["color_inputs"]),
                dmc.Button("Применить", id=self.ids["apply_colors"]),
            ],
        )

    def _help_modal(self):
        default_type = self.help_default_type
        content, title = render_instruction(default_type)
        return dmc.Modal(
            id=self.ids["help_modal"],
            title=title,
            opened=False,
            size="lg",
            zIndex=1000,
            centered=True,
            className="graph-help-modal",
            children=[
                dmc.Select(
                    id=self.ids["help_type"],
                    label="Тип графика",
                    data=self.help_options,
                    value=default_type,
                    allowDeselect=False,
                    clearable=False,
                    comboboxProps={"shadow": "md", "withinPortal": True, "zIndex": 1100},
                    mb="sm",
                ),
                dmc.ScrollArea(
                    dmc.Stack(id=self.ids["help_content"], gap="xs", children=content),
                    h="min(58dvh, 540px)",
                    type="auto",
                    scrollbars="y",
                    offsetScrollbars="y",
                    scrollbarSize=8,
                    className="graph-help-scroll",
                ),
                dmc.Group(
                    dmc.Button(
                        "Закрыть",
                        id=self.ids["help_close"],
                        variant="default",
                        size="xs",
                    ),
                    justify="flex-end",
                    mt="md",
                ),
            ],
        )

    def _service_components(self):
        children = [
            dcc.Store(id=self.ids["view_revision"], data=0),
            dcc.Store(id=self.ids["custom_colors"], data={}),
            dcc.Store(id=self.ids["sync"]),
            html.Button("download-html", id=self.ids["download_html"]),
            html.Button("update", id=self.ids["update"]),
            html.Button("copy-png", id=self.ids["copy_png"]),
            html.Button("save-png", id=self.ids["save_png"]),
            html.Button("clear-graph", id=self.ids["clear"]),
            dcc.Download(id=self.ids["download_component"]),
        ]
        if self.include_color_controls:
            children.append(html.Button("change-colors", id=self.ids["change_colors"]))
        return html.Div(
            children,
            className="graph-workspace-services",
            style={"display": "none"},
            **{"aria-hidden": "true"},
        )

    def render(self):
        """Return the complete component tree, not only its Plotly paper."""
        children = [self._render_paper()]
        if self.settings_panel is not None:
            children.append(self.settings_panel.render())
        if self.include_color_controls:
            children.append(self._color_modal())
        children.append(self._help_modal())
        children.append(self._service_components())
        return html.Div(
            children,
            id=self.component_id,
            className="graph-workspace-component",
        )

    def field_id(self, key: str) -> str:
        """Return a field control ID by its semantic key (x, y, color, ...)."""
        try:
            return self.field_ids_by_key[key]
        except KeyError as error:
            raise KeyError(f"GraphWorkspace has no field {key!r}") from error

    def settings_control_id(self, key: str) -> str:
        """Return the external settings control ID owned by this workspace."""
        if self.settings_panel is None:
            raise RuntimeError("GraphWorkspace has no settings panel")
        return _component_id(self.settings_panel.controls[key])

    def settings_id(self, legacy_id: str) -> str:
        """Return an instance-scoped ID for a settings panel component."""
        if self.settings_panel is None:
            raise RuntimeError("GraphWorkspace has no settings panel")
        return self.settings_panel.component_id(legacy_id)

    def register_callbacks(self, app):
        """Register UI/export callbacks owned by this workspace exactly once."""
        if self._callbacks_registered:
            return
        self._callbacks_registered = True

        self._register_clear_callback(app)
        self._register_zone_labels_callback(app)
        self._register_export_callbacks(app)
        self._register_help_callbacks(app)
        if self.settings_panel is not None:
            self._register_settings_callbacks(app)
        if self.include_color_controls:
            self._register_color_callbacks(app)

    def figure_callback(self, app, *dependencies, prevent_initial_call=True):
        """Return a decorator that connects a dashboard figure builder.

        A builder returns ``(figure, notifications)``. This keeps data-domain
        logic replaceable while the workspace owns its public Dash outputs.
        """
        return app.callback(
            Output(self.graph_id, "figure"),
            Output(self.notifications_id, "sendNotifications", allow_duplicate=True),
            *dependencies,
            prevent_initial_call=prevent_initial_call,
        )

    def _register_clear_callback(self, app):
        outputs = [
            Output(self.field_ids[field["target"]], "value")
            for field in self.fields
        ]
        outputs.append(Output(self.ids["view_revision"], "data"))
        empty_values = [
            [] if field.get("mode") == "append" else None
            for field in self.fields
        ]
        recovery_script = _plotly_recovery_script(self.graph_id)

        app.clientside_callback(
            f"""
            function(nClicks, currentRevision) {{
                if (!nClicks) {{
                    return Array({len(outputs)}).fill(window.dash_clientside.no_update);
                }}
{recovery_script}
                var values = {json.dumps(empty_values, ensure_ascii=False)};
                values.push((Number(currentRevision) || 0) + 1);
                return values;
            }}
            """,
            *outputs,
            Input(self.ids["clear"], "n_clicks"),
            State(self.ids["view_revision"], "data"),
            prevent_initial_call=True,
        )

    def _register_zone_labels_callback(self, app):
        targets = [self.field_ids[field["target"]] for field in self.fields]
        inputs = [Input(target, "value") for target in targets]

        app.clientside_callback(
            f"""
            function() {{
                var targets = {json.dumps(targets, ensure_ascii=False)};
                var values = Array.prototype.slice.call(arguments);
                var byTarget = {{}};
                targets.forEach(function(target, index) {{ byTarget[target] = values[index]; }});
                var workspace = document.getElementById({json.dumps(self.workspace_id)});
                if (!workspace) return window.dash_clientside.no_update;

                workspace.querySelectorAll('.graph-drop-zone').forEach(function(zone) {{
                    var value = byTarget[zone.getAttribute('data-drop-target')];
                    var hasValue = Array.isArray(value) ? value.length > 0 : Boolean(value);
                    var valueElement = zone.querySelector('.graph-drop-zone-value');
                    if (valueElement) {{
                        valueElement.textContent = Array.isArray(value)
                            ? (value.join(', ') || 'Не выбрано')
                            : (value || 'Не выбрано');
                    }}
                    zone.setAttribute('data-current-value', JSON.stringify(value ?? null));
                    zone.classList.toggle('has-value', hasValue);
                }});
                return Date.now();
            }}
            """,
            Output(self.ids["sync"], "data"),
            *inputs,
            prevent_initial_call="initial_duplicate",
        )

    def _register_settings_callbacks(self, app):
        app.clientside_callback(
            """
            function (chartType) {
                var pointTypes = ["Scatter", "3D_Scatter", "Polar"];
                var points = pointTypes.indexOf(chartType) !== -1;
                var bars = chartType === "Bar" || chartType === "Hist";
                var barOnly = chartType === "Bar";
                var pie = chartType === "Pie";
                var hasSpecific = points || bars || pie;
                var labels = {
                    "Scatter": "Scatter",
                    "3D_Scatter": "3D Scatter",
                    "Polar": "Треугольный график",
                    "Bar": "Bar",
                    "Hist": "Histogram",
                    "Pie": "Pie"
                };
                function visible(show) {
                    return {display: show ? "block" : "none"};
                }
                return [
                    "Настройки: " + (labels[chartType] || chartType || "график"),
                    visible(points),
                    visible(bars),
                    visible(barOnly),
                    visible(pie),
                    visible(!hasSpecific)
                ];
            }
            """,
            Output(self.settings_id("graph-settings-specific-title"), "children"),
            Output(self.settings_id("graph-settings-specific-points"), "style"),
            Output(self.settings_id("graph-settings-specific-bars"), "style"),
            Output(self.settings_id("graph-settings-specific-bar-only"), "style"),
            Output(self.settings_id("graph-settings-specific-pie"), "style"),
            Output(self.settings_id("graph-settings-specific-empty"), "style"),
            Input(self.chart_type_id, "value"),
        )

        defaults = {
            self.settings_control_id("theme"): {"value": "plotly"},
            self.settings_id("InputSizePlot"): {"value": 750},
            self.settings_id("InputSizePlotW"): {"value": None},
            self.settings_id("font-size-xaxis"): {"value": 14},
            self.settings_id("font-size-yaxis"): {"value": 14},
            self.settings_id("font-size-title"): {"value": 16},
            self.settings_id("font-size-ticks"): {"value": 12},
            self.settings_id("tick-step-xaxis"): {"value": 0},
            self.settings_id("tick-step-yaxis"): {"value": 0},
            self.settings_control_id("text_position"): {"value": "middle center"},
            self.settings_control_id("category_axis"): {"value": "auto"},
            self.settings_control_id("category_order"): {"value": "total ascending"},
            self.settings_control_id("bar_mode"): {"value": "overlay"},
            self.settings_control_id("bar_aggregation"): {"value": "sum"},
            self.settings_control_id("pie_aggregation"): {"value": "sum"},
            self.settings_control_id("legend_position"): {"value": "top-right-outside"},
            self.settings_control_id("legend_order"): {"value": "alphabetical"},
            self.settings_control_id("legend_custom_order"): {"value": ""},
            self.settings_control_id("bubble"): {"checked": True},
            self.settings_id("InputMaxSizeBubble"): {"value": 30},
            self.settings_control_id("bar_labels"): {"checked": True},
        }

        app.clientside_callback(
            f"""
            function(nClicks) {{
                if (!nClicks) return window.dash_clientside.no_update;
                var defaults = {json.dumps(defaults, ensure_ascii=False)};
                Object.keys(defaults).forEach(function(componentId) {{
                    window.dash_clientside.set_props(componentId, defaults[componentId]);
                }});
                return {{clicks: nClicks, resetAt: Date.now()}};
            }}
            """,
            Output(self.settings_id("graph-settings-reset-state"), "data"),
            Input(self.settings_id("graph-settings-reset"), "n_clicks"),
            prevent_initial_call=True,
        )

        app.clientside_callback(
            """
            function(height, width) {
                function pixelSize(value, fallback) {
                    var number = Number(value);
                    return Number.isFinite(number) && number > 0
                        ? Math.round(number) + 'px'
                        : fallback;
                }
                return {
                    height: pixelSize(height, '750px'),
                    width: pixelSize(width, '100%')
                };
            }
            """,
            Output(self.paper_id, "style"),
            Input(self.settings_id("InputSizePlot"), "value"),
            Input(self.settings_id("InputSizePlotW"), "value"),
        )

    def _register_export_callbacks(self, app):
        export_field_ids = [
            self.field_id(key)
            for key in ("x", "y")
            if key in self.field_ids_by_key
        ]
        export_states = [State(component_id, "value") for component_id in export_field_ids]

        @app.callback(
            Output(self.ids["download_component"], "data"),
            Output(self.notifications_id, "sendNotifications", allow_duplicate=True),
            Input(self.ids["download_html"], "n_clicks"),
            State(self.graph_id, "figure"),
            *export_states,
            State(self.chart_type_id, "value"),
            prevent_initial_call=True,
        )
        def download_html(n_clicks, figure, *field_values_and_chart_type):
            if not n_clicks or not figure:
                raise PreventUpdate
            try:
                *field_values, chart_type = field_values_and_chart_type
                fig = go.Figure(figure)
                axis_label = " vs ".join(str(value) for value in field_values if value)
                filename_parts = [part for part in (axis_label, chart_type) if part]
                filename = (
                    " ".join(filename_parts) + ".html"
                    if filename_parts
                    else f"{self.graph_id}.html"
                )
                return {
                    "content": fig.to_html(include_plotlyjs="cdn"),
                    "filename": filename,
                    "type": "text/html",
                }, []
            except Exception as error:
                return no_update, _make_error_notif(f"Ошибка скачивания: {error}")

        def png_callback(method: str, success_title: str, success_message: str, error_title: str):
            return f"""
            function(nClicks, figure) {{
                if (!nClicks || !figure) {{
                    throw window.dash_clientside.PreventUpdate;
                }}
                function notification(title, message, color) {{
                    return [{{
                        id: crypto.randomUUID(), title: title, message: message,
                        color: color, action: 'show', autoClose: 4500
                    }}];
                }}
                if (!window.graphPng) {{
                    return notification({json.dumps(error_title)}, 'Модуль экспорта не загружен.', 'red');
                }}
                return window.graphPng[{json.dumps(method)}]({json.dumps(self.graph_id)}).then(
                    () => notification({json.dumps(success_title)}, {json.dumps(success_message)}, 'green'),
                    (error) => {{
                        console.error('PNG export error:', error);
                        return notification({json.dumps(error_title)}, error.message || 'Ошибка экспорта.', 'red');
                    }}
                );
            }}
            """

        app.clientside_callback(
            png_callback(
                "copyToClipboard",
                "PNG скопирован",
                "Изображение помещено в буфер обмена.",
                "PNG не скопирован",
            ),
            Output(self.notifications_id, "sendNotifications", allow_duplicate=True),
            Input(self.ids["copy_png"], "n_clicks"),
            State(self.graph_id, "figure"),
            prevent_initial_call=True,
        )
        app.clientside_callback(
            png_callback(
                "saveToFile",
                "PNG сохранён",
                "Файл с графиком передан в загрузки.",
                "PNG не сохранён",
            ),
            Output(self.notifications_id, "sendNotifications", allow_duplicate=True),
            Input(self.ids["save_png"], "n_clicks"),
            State(self.graph_id, "figure"),
            prevent_initial_call=True,
        )

    def _register_help_callbacks(self, app):
        @app.callback(
            Output(self.ids["help_modal"], "opened", allow_duplicate=True),
            Output(self.ids["help_type"], "value"),
            Input(self.ids["help"], "n_clicks"),
            State(self.chart_type_id, "value"),
            prevent_initial_call=True,
        )
        def open_help(n_clicks, chart_type):
            if not n_clicks:
                raise PreventUpdate
            known = chart_type if chart_type in GRAPH_INSTRUCTIONS else self.help_default_type
            return True, known

        @app.callback(
            Output(self.ids["help_content"], "children"),
            Output(self.ids["help_modal"], "title"),
            Input(self.ids["help_type"], "value"),
        )
        def render_help(chart_type):
            return render_instruction(chart_type)

        app.clientside_callback(
            """
            function(nClicks) {
                if (!nClicks) {
                    return window.dash_clientside.no_update;
                }
                return false;
            }
            """,
            Output(self.ids["help_modal"], "opened", allow_duplicate=True),
            Input(self.ids["help_close"], "n_clicks"),
            prevent_initial_call=True,
        )

    def _register_color_callbacks(self, app):
        picker_type = f"{self.graph_id}-color-picker"
        preview_type = f"{self.graph_id}-color-preview"
        theme_id = self.settings_control_id("theme")

        @app.callback(
            Output(self.ids["color_modal"], "opened", allow_duplicate=True),
            Output(self.ids["color_inputs"], "children"),
            Input(self.ids["change_colors"], "n_clicks"),
            Input(self.ids["color_mode"], "checked"),
            State(self.graph_id, "figure"),
            State(theme_id, "value"),
            State(self.ids["custom_colors"], "data"),
            prevent_initial_call=True,
        )
        def open_color_dialog(_n_clicks, manual_mode, figure, selected_style, custom_colors):
            if not figure or "data" not in figure:
                raise PreventUpdate

            traces = figure["data"]
            use_dropdown = not manual_mode and len(traces) <= COLOR_THRESHOLD
            if selected_style == "seaborn_custom":
                style_colors = pio.templates[selected_style].layout.colorway
            else:
                style_colors = getattr(
                    px.colors.qualitative,
                    selected_style or "Plotly",
                    px.colors.qualitative.Plotly,
                )

            color_inputs = []
            for index_number, trace in enumerate(traces):
                index = str(index_number)
                name = trace.get("name", f"Категория {index_number + 1}")
                current_color = (custom_colors or {}).get(
                    index,
                    trace.get("marker", {}).get(
                        "color",
                        style_colors[index_number % len(style_colors)],
                    ),
                )
                preview_id = {"type": preview_type, "index": index}

                if use_dropdown:
                    input_control = dmc.Group(
                        [
                            dcc.Dropdown(
                                id={"type": picker_type, "index": index},
                                value=(
                                    current_color
                                    if current_color in style_colors
                                    else style_colors[index_number % len(style_colors)]
                                ),
                                options=[
                                    {"label": f"Цвет {position + 1} ({color})", "value": color}
                                    for position, color in enumerate(style_colors)
                                ],
                                clearable=False,
                                style={"width": 300},
                            ),
                            html.Div(
                                id=preview_id,
                                style={
                                    "backgroundColor": current_color,
                                    "width": "20px",
                                    "height": "20px",
                                    "border": "1px solid #ccc",
                                    "marginLeft": "5px",
                                },
                            ),
                        ]
                    )
                else:
                    input_control = dmc.ColorInput(
                        id={"type": picker_type, "index": index},
                        value=current_color,
                        format="hex",
                    )
                color_inputs.append(
                    dmc.Group([dmc.Text(name, style={"width": 150}), input_control])
                )
            return True, color_inputs

        @app.callback(
            Output(self.graph_id, "figure", allow_duplicate=True),
            Output(self.ids["color_modal"], "opened", allow_duplicate=True),
            Output(self.ids["custom_colors"], "data"),
            Output(self.notifications_id, "sendNotifications", allow_duplicate=True),
            Input(self.ids["apply_colors"], "n_clicks"),
            State(self.graph_id, "figure"),
            State({"type": picker_type, "index": ALL}, "id"),
            State({"type": picker_type, "index": ALL}, "value"),
            State(self.ids["custom_colors"], "data"),
            prevent_initial_call=True,
        )
        def apply_custom_colors(_n_clicks, figure, picker_ids, values, custom_colors):
            if not figure or "data" not in figure:
                raise PreventUpdate
            try:
                updated = (custom_colors or {}).copy()
                updated.update({item["index"]: value for item, value in zip(picker_ids, values)})
                result = apply_custom_colors_safely(go.Figure(figure), updated)
                return result, False, updated, []
            except Exception as error:
                logger.error("Ошибка при применении цветов: %s", error, exc_info=True)
                return (
                    figure,
                    False,
                    custom_colors,
                    _make_error_notif("Не удалось применить цвета. График остаётся без изменений."),
                )

        @app.callback(
            Output({"type": preview_type, "index": MATCH}, "style"),
            Input({"type": picker_type, "index": MATCH}, "value"),
            prevent_initial_call=True,
        )
        def update_preview_color(selected_color):
            return {
                "backgroundColor": selected_color,
                "width": "20px",
                "height": "20px",
                "border": "1px solid #ccc",
                "marginLeft": "5px",
            }
