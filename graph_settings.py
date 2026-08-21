# -*- coding: utf-8 -*-
"""Extensible, instance-scoped property inspector for GraphWorkspace."""

from __future__ import annotations

from collections.abc import Mapping

from dash import dcc, html
import dash_mantine_components as dmc


REQUIRED_CONTROLS = {
    "theme",
    "render_mode",
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
    "graph-settings-popover",
    "graph-settings-close",
    "graph-settings-common",
    "graph-settings-specific",
    "graph-settings-specific-title",
    "graph-settings-specific-points",
    "graph-settings-specific-bars",
    "graph-settings-specific-bar-only",
    "graph-settings-specific-pie",
    "graph-settings-specific-empty",
    "graph-settings-tabs",
    "graph-settings-reset-state",
    "graph-settings-reset",
    "graph-settings-close-on-outside",
    "InputSizePlot",
    "InputSizePlotW",
    "font-size-xaxis",
    "font-size-yaxis",
    "tick-step-xaxis",
    "tick-step-yaxis",
    "font-size-title",
    "font-size-ticks",
    "InputMaxSizeBubble",
    "InputMarkerSize",
)

SETTINGS_COMBOBOX_Z_INDEX = 10030


class GraphSettingsPanel:
    """Build a scalable, single-level graph settings inspector."""

    def __init__(self, controls: Mapping[str, object], ids: Mapping[str, str] | None = None):
        self.controls = controls
        self.width = 340
        self._namespace = None
        self._explicit_ids = set((ids or {}).keys())
        self.ids = {component_id: component_id for component_id in SETTINGS_COMPONENT_IDS}
        self.ids.update(ids or {})
        missing = REQUIRED_CONTROLS.difference(controls)
        if missing:
            raise ValueError(f"Missing graph settings controls: {sorted(missing)}")

        # Mantine renders Select menus in a body-level portal. Its default
        # layer (300) is below our movable settings window (10020), making a
        # menu look as though it did not open. Apply the fix to every current
        # and future combobox control supplied to this panel.
        for control in controls.values():
            if hasattr(control, "comboboxProps"):
                combobox_props = dict(getattr(control, "comboboxProps", None) or {})
                combobox_props["zIndex"] = SETTINGS_COMBOBOX_Z_INDEX
                control.comboboxProps = combobox_props

    def bind_namespace(self, namespace: str):
        """Bind internal IDs to exactly one GraphWorkspace instance."""
        namespace = str(namespace).strip()
        if not namespace:
            raise ValueError("GraphSettingsPanel namespace must not be empty")
        if self._namespace and self._namespace != namespace:
            raise ValueError(
                f"GraphSettingsPanel is already bound to {self._namespace!r}"
            )
        if self._namespace == namespace:
            return self
        self._namespace = namespace
        for legacy_id in SETTINGS_COMPONENT_IDS:
            if legacy_id not in self._explicit_ids:
                self.ids[legacy_id] = f"{namespace}-{legacy_id}"
        return self

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
                    spacing=6,
                    verticalSpacing=6,
                    mt=8,
                    className="graph-settings-quick-grid",
                ),
                html.Div(
                    [
                        dmc.Text("Рендер", size="10px", fw=600, mb=4),
                        self.controls["render_mode"],
                    ],
                    className="graph-settings-control graph-settings-render-mode",
                ),
            ],
            p=8,
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
                    ],
                    cols=2,
                    spacing="xs",
                    verticalSpacing="xs",
                    className="graph-settings-grid",
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

    def _specific_settings(self):
        points = html.Div(
            [
                self._section_intro(
                    "Точки и подписи",
                    "Параметры Scatter, 3D Scatter и треугольного графика",
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
                            "Макс. размер пузыря",
                            30,
                            min=1,
                            max=100,
                            step=5,
                        ),
                        html.Div(
                            self._number_input(
                                "InputMarkerSize",
                                "Размер маркера, px",
                                8,
                                min=1,
                                max=100,
                                step=1,
                                decimalScale=1,
                            ),
                            className="graph-settings-span-2",
                        ),
                    ],
                    cols=2,
                    spacing="xs",
                    verticalSpacing="xs",
                    className="graph-settings-grid",
                ),
                dmc.Divider(my="sm", label="Положение подписей", labelPosition="left"),
                html.Div(self.controls["text_position"], className="graph-settings-control"),
            ],
            id=self.component_id("graph-settings-specific-points"),
            className="graph-settings-specific-group",
        )

        bars = html.Div(
            [
                self._section_intro(
                    "Столбцы и гистограммы",
                    "Режим отображения серий и параметры агрегации Bar",
                    "list",
                ),
                html.Div(self.controls["bar_mode"], className="graph-settings-control"),
                html.Div(
                    [
                        dmc.Divider(my="sm", label="Только Bar", labelPosition="left"),
                        html.Div(self.controls["bar_aggregation"], className="graph-settings-control"),
                        self._feature_card(
                            self.controls["bar_labels"],
                            "Показывает числовые значения непосредственно на столбцах Bar.",
                        ),
                    ],
                    id=self.component_id("graph-settings-specific-bar-only"),
                ),
            ],
            id=self.component_id("graph-settings-specific-bars"),
            className="graph-settings-specific-group",
        )

        pie = html.Div(
            [
                self._section_intro(
                    "Круговая диаграмма",
                    "Способ объединения значений внутри каждого сектора",
                    "scatter-chart",
                ),
                html.Div(self.controls["pie_aggregation"], className="graph-settings-control"),
            ],
            id=self.component_id("graph-settings-specific-pie"),
            className="graph-settings-specific-group",
        )

        empty = html.Div(
            [
                self._icon("info", 16),
                dmc.Text("Для этого типа графика специальных настроек пока нет.", size="xs"),
                dmc.Text(
                    "Общие параметры доступны через контекстное меню графика.",
                    size="10px",
                    c="dimmed",
                ),
            ],
            id=self.component_id("graph-settings-specific-empty"),
            className="graph-settings-specific-empty",
        )
        return [points, bars, pie, empty]

    def render(self):
        return self._settings_body()

    def _settings_body(self):
        tabs = dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.TabsTab("Оси", value="axes", leftSection=self._icon("move-horizontal", 15)),
                        dmc.TabsTab("Подписи", value="labels", leftSection=self._icon("type", 15)),
                        dmc.TabsTab("Легенда", value="legend", leftSection=self._icon("list", 15)),
                    ],
                    grow=True,
                    className="graph-settings-tabs-list",
                ),
                self._axes_panel(),
                self._labels_panel(),
                self._legend_panel(),
            ],
            id=self.component_id("graph-settings-tabs"),
            value="axes",
            keepMounted=True,
            persistence=True,
            persistence_type="local",
            className="graph-settings-tabs",
        )

        return html.Section(
            id=self.component_id("graph-settings-popover"),
            className="graph-settings-popover",
            role="dialog",
            **{
                "aria-hidden": "true",
                "data-close-on-outside-id": self.component_id("graph-settings-close-on-outside"),
            },
            children=[
                dcc.Store(id=self.component_id("graph-settings-reset-state")),
                html.Header(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "⠿",
                                    className="graph-settings-drag-handle",
                                    title="Перетащить окно",
                                    **{"aria-hidden": "true"},
                                ),
                                html.Div(
                                    [
                                        dmc.Text("Общие настройки", fw=700, size="sm"),
                                        dmc.Text("Параметры, общие для всех типов", size="10px", c="dimmed"),
                                    ],
                                    className="graph-settings-heading graph-settings-heading--common",
                                ),
                                html.Div(
                                    [
                                        dmc.Text(
                                            "Настройки типа",
                                            id=self.component_id("graph-settings-specific-title"),
                                            fw=700,
                                            size="sm",
                                        ),
                                        dmc.Text("Только параметры выбранного графика", size="10px", c="dimmed"),
                                    ],
                                    className="graph-settings-heading graph-settings-heading--specific",
                                ),
                            ],
                            className="graph-settings-heading-wrap",
                        ),
                        html.Button(
                            "×",
                            id=self.component_id("graph-settings-close"),
                            type="button",
                            className="graph-settings-popover-close",
                            title="Закрыть настройки",
                            **{"aria-label": "Закрыть настройки"},
                        ),
                    ],
                    className="graph-settings-popover-header",
                ),
                html.Div(
                    [self._quick_settings(), tabs],
                    id=self.component_id("graph-settings-common"),
                    className="graph-settings-scroll graph-settings-common",
                ),
                html.Div(
                    self._specific_settings(),
                    id=self.component_id("graph-settings-specific"),
                    className="graph-settings-scroll graph-settings-specific",
                ),
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
                            dmc.Checkbox(
                                id=self.component_id("graph-settings-close-on-outside"),
                                label="Закрывать при клике вне",
                                checked=False,
                                size="xs",
                                persistence=True,
                                persistence_type="local",
                            ),
                            className="graph-settings-footer-options",
                        ),
                    ],
                    className="graph-settings-footer",
                ),
            ],
        )
