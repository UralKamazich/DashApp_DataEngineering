# -*- coding: utf-8 -*-
"""Extensible property inspector for the main graph."""

from __future__ import annotations

from collections.abc import Mapping

from dash import dcc, html
import dash_mantine_components as dmc

from slide_panel import SlidePanel


REQUIRED_CONTROLS = {
    "theme",
    "bubble",
    "bar_labels",
    "text_position",
    "category_axis",
    "category_order",
    "bar_mode",
    "bar_aggregation",
    "pie_aggregation",
    "legend_position",
    "legend_order",
    "legend_custom_order",
}

SETTINGS_COMPONENT_IDS = (
    "drawer-simple",
    "drawer-simple-tab",
    "drawer-simple-open-state",
    "graph-settings-tabs",
    "graph-settings-panel",
    "graph-settings-reset-state",
    "graph-settings-reset",
    "graph-settings-close-on-outside",
    "graph-settings-shift-plot",
    "graph-settings-outside-close-store",
    "InputSizePlot",
    "InputSizePlotW",
    "font-size-xaxis",
    "font-size-yaxis",
    "tick-step-xaxis",
    "tick-step-yaxis",
    "font-size-title",
    "font-size-ticks",
    "InputMaxSizeBubble",
)


class GraphSettingsPanel:
    """Build a scalable, single-level graph settings inspector."""

    def __init__(self, controls: Mapping[str, object], ids: Mapping[str, str] | None = None):
        self.controls = controls
        self.ids = {component_id: component_id for component_id in SETTINGS_COMPONENT_IDS}
        self.ids.update(ids or {})
        missing = REQUIRED_CONTROLS.difference(controls)
        if missing:
            raise ValueError(f"Missing graph settings controls: {sorted(missing)}")
        self.slide = SlidePanel(
            root_id=self.component_id("drawer-simple"),
            tab_id=self.component_id("drawer-simple-tab"),
            state_id=self.component_id("drawer-simple-open-state"),
            side="right",
            mode="overlay",
            width=340,
            tab_icon=self._icon("sliders-horizontal", 15),
            tab_label="Настройки",
            tab_title="Открыть настройки графика",
            tab_style={"bottom": "10px"},
            extra_root_classes=("graph-settings-content",),
            content=self._settings_body,
            extra_stores=[
                dcc.Store(id=self.component_id("graph-settings-outside-close-store"), data=False),
            ],
        )

    def component_id(self, legacy_id: str) -> str:
        """Return the instance-specific id for an internal settings control."""
        return self.ids.get(legacy_id, legacy_id)

    @staticmethod
    def _icon(name: str, size: int = 16):
        # Offline-safe symbols: unlike remote icon packs, these also render when
        # Electron has no network access.
        symbols = {
            "sliders-horizontal": "⌘",
            "move-horizontal": "↔",
            "type": "T",
            "list": "≡",
            "scatter-chart": "∴",
            "zap": "⚡",
            "circle-check": "✓",
            "rotate-ccw": "↺",
            "info": "i",
        }
        return html.Span(
            symbols.get(name, "•"),
            className=f"graph-settings-icon graph-settings-icon--{name}",
            style={"width": f"{size}px", "height": f"{size}px", "fontSize": f"{size}px"},
            **{"aria-hidden": "true"},
        )

    def _number_input(self, component_id: str, label: str, value, **kwargs):
        return dmc.NumberInput(
            id=self.component_id(component_id),
            label=label,
            value=value,
            size="xs",
            debounce=True,
            persistence=True,
            persistence_type="local",
            className="graph-settings-control",
            **kwargs,
        )

    @classmethod
    def _section_intro(cls, title: str, description: str, icon: str):
        return dmc.Group(
            [
                html.Div(cls._icon(icon, 13), className="graph-settings-section-icon"),
                html.Div(
                    [
                        dmc.Text(title, fw=650, size="xs"),
                        dmc.Text(description, size="10px", c="dimmed"),
                    ]
                ),
            ],
            gap="xs",
            align="flex-start",
            wrap="nowrap",
            className="graph-settings-section-intro",
        )

    @staticmethod
    def _feature_card(control, description: str):
        return html.Div(
            [control, dmc.Text(description, size="10px", c="dimmed")],
            className="graph-settings-feature-card",
        )

    def _quick_settings(self):
        return dmc.Paper(
            [
                dmc.Group(
                    [
                        html.Div(
                            [
                                dmc.Text("Быстрый доступ", fw=650, size="xs"),
                                dmc.Text(
                                    "Самые частые параметры всегда под рукой",
                                    size="10px",
                                    c="dimmed",
                                ),
                            ]
                        ),
                        dmc.Badge(
                            "Сразу",
                            color="teal",
                            variant="light",
                            size="xs",
                            leftSection=self._icon("zap", 12),
                        ),
                    ],
                    justify="space-between",
                    align="flex-start",
                    gap="xs",
                    wrap="nowrap",
                ),
                dmc.SimpleGrid(
                    [
                        html.Div(
                            self.controls["theme"],
                            className="graph-settings-control graph-settings-quick-theme",
                        ),
                        self._number_input(
                            "InputSizePlot",
                            "Высота",
                            750,
                            min=50,
                            max=20000,
                            step=50,
                            suffix=" px",
                        ),
                        self._number_input(
                            "InputSizePlotW",
                            "Ширина",
                            None,
                            min=50,
                            max=20000,
                            step=50,
                            suffix=" px",
                            placeholder="Авто",
                        ),
                    ],
                    cols=3,
                    spacing="xs",
                    verticalSpacing="xs",
                    mt="sm",
                    className="graph-settings-quick-grid",
                ),
            ],
            p="xs",
            radius="md",
            withBorder=True,
            className="graph-settings-quick",
        )

    def _axes_panel(self):
        return dmc.TabsPanel(
            [
                self._section_intro(
                    "Оси и категории",
                    "Размеры шрифта, интервалы тиков и порядок категорий",
                    "move-horizontal",
                ),
                dmc.SimpleGrid(
                    [
                        self._number_input("font-size-xaxis", "Подпись оси X", 14, min=6, max=48, step=1),
                        self._number_input("font-size-yaxis", "Подпись оси Y", 14, min=6, max=48, step=1),
                        self._number_input(
                            "tick-step-xaxis", "Шаг тиков X", 0, min=0, step=0.1, decimalScale=2
                        ),
                        self._number_input(
                            "tick-step-yaxis", "Шаг тиков Y", 0, min=0, step=0.1, decimalScale=2
                        ),
                    ],
                    cols=2,
                    spacing="xs",
                    verticalSpacing="xs",
                    className="graph-settings-grid",
                ),
                dmc.Divider(my="sm", label="Категориальные оси", labelPosition="left"),
                dmc.SimpleGrid(
                    [
                        html.Div(self.controls["category_axis"], className="graph-settings-control"),
                        html.Div(self.controls["category_order"], className="graph-settings-control"),
                    ],
                    cols=2,
                    spacing="xs",
                    className="graph-settings-grid",
                ),
            ],
            value="axes",
            className="graph-settings-tab-panel",
        )

    def _labels_panel(self):
        return dmc.TabsPanel(
            [
                self._section_intro(
                    "Текст и подписи",
                    "Размер заголовка, подписи данных и их положение",
                    "type",
                ),
                dmc.SimpleGrid(
                    [
                        self._number_input("font-size-title", "Заголовок", 16, min=6, max=48, step=1),
                        self._number_input("font-size-ticks", "Подписи данных", 12, min=6, max=48, step=1),
                        html.Div(self.controls["text_position"], className="graph-settings-control graph-settings-span-2"),
                    ],
                    cols=2,
                    spacing="xs",
                    verticalSpacing="xs",
                    className="graph-settings-grid",
                ),
                dmc.Divider(my="sm"),
                self._feature_card(
                    self.controls["bar_labels"],
                    "Показывает числовые значения непосредственно на столбцах Bar.",
                ),
            ],
            value="labels",
            className="graph-settings-tab-panel",
        )

    def _legend_panel(self):
        return dmc.TabsPanel(
            [
                self._section_intro(
                    "Легенда",
                    "Положение и порядок серий без изменения данных",
                    "list",
                ),
                dmc.SimpleGrid(
                    [
                        html.Div(self.controls["legend_position"], className="graph-settings-control"),
                        html.Div(self.controls["legend_order"], className="graph-settings-control"),
                        html.Div(
                            self.controls["legend_custom_order"],
                            className="graph-settings-control graph-settings-span-2",
                        ),
                    ],
                    cols=2,
                    spacing="xs",
                    verticalSpacing="xs",
                    className="graph-settings-grid",
                ),
                dmc.Alert(
                    "Свой порядок применяется, когда в поле выше выбран режим «Пользовательский».",
                    color="gray",
                    variant="light",
                    icon=self._icon("info", 14),
                    mt="sm",
                    className="graph-settings-note",
                ),
            ],
            value="legend",
            className="graph-settings-tab-panel",
        )

    def _series_panel(self):
        return dmc.TabsPanel(
            [
                self._section_intro(
                    "Серии и маркеры",
                    "Отображение точек, столбцов и пузырьковых диаграмм",
                    "scatter-chart",
                ),
                dmc.SimpleGrid(
                    [
                        html.Div(
                            self.controls["bubble"],
                            className="graph-settings-control graph-settings-bubbles",
                        ),
                        self._number_input(
                            "InputMaxSizeBubble",
                            "Максимальный размер маркера",
                            30,
                            min=1,
                            max=100,
                            step=5,
                        ),
                    ],
                    cols=2,
                    spacing="xs",
                    verticalSpacing="xs",
                    className="graph-settings-grid",
                ),
                dmc.Divider(my="sm", label="Столбцы и гистограммы", labelPosition="left"),
                dmc.SimpleGrid(
                    [
                        html.Div(self.controls["bar_mode"], className="graph-settings-control"),
                        html.Div(self.controls["bar_aggregation"], className="graph-settings-control"),
                    ],
                    cols=2,
                    spacing="xs",
                    verticalSpacing="xs",
                    className="graph-settings-grid",
                ),
                dmc.Divider(my="sm", label="Круговая диаграмма", labelPosition="left"),
                html.Div(self.controls["pie_aggregation"], className="graph-settings-control"),
            ],
            value="series",
            className="graph-settings-tab-panel",
        )

    def render(self):
        return self.slide.render()

    def _settings_body(self):
        tabs = dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.TabsTab("Оси", value="axes", leftSection=self._icon("move-horizontal", 15)),
                        dmc.TabsTab("Подписи", value="labels", leftSection=self._icon("type", 15)),
                        dmc.TabsTab("Легенда", value="legend", leftSection=self._icon("list", 15)),
                        dmc.TabsTab("Серии", value="series", leftSection=self._icon("scatter-chart", 15)),
                    ],
                    grow=True,
                    className="graph-settings-tabs-list",
                ),
                self._axes_panel(),
                self._labels_panel(),
                self._legend_panel(),
                self._series_panel(),
            ],
            id=self.component_id("graph-settings-tabs"),
            value="axes",
            keepMounted=True,
            persistence=True,
            persistence_type="local",
            className="graph-settings-tabs",
        )

        return [
            html.Div(
                id=self.component_id("graph-settings-panel"),
                className="graph-settings-panel-inner",
                children=[
                    dcc.Store(id=self.component_id("graph-settings-reset-state")),
                    html.Div([self._quick_settings(), tabs], className="graph-settings-scroll"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            self._icon("circle-check", 13),
                                            dmc.Text("Применяется автоматически", size="10px", c="dimmed"),
                                        ],
                                        className="graph-settings-auto-status",
                                    ),
                                    dmc.Button(
                                        "Сбросить",
                                        id=self.component_id("graph-settings-reset"),
                                        variant="subtle",
                                        color="gray",
                                        size="xs",
                                        leftSection=self._icon("rotate-ccw", 12),
                                    ),
                                ],
                                className="graph-settings-footer-main",
                            ),
                            html.Div(
                                [
                                    dmc.Checkbox(
                                        id=self.component_id("graph-settings-shift-plot"),
                                        label="Сдвигать график при открытии",
                                        checked=False,
                                        size="xs",
                                        persistence=True,
                                    ),
                                    dmc.Checkbox(
                                        id=self.component_id("graph-settings-close-on-outside"),
                                        label="Закрывать при клике вне",
                                        checked=False,
                                        size="xs",
                                        persistence=True,
                                    ),
                                ],
                                className="graph-settings-footer-options",
                            ),
                        ],
                        className="graph-settings-footer",
                    ),
                ],
            ),
        ]
