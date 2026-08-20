# -*- coding: utf-8 -*-
"""Унифицированная выезжающая панель с прикреплённым ярлычком-ручкой.

Механика одна и та же для всех панелей (датасет, фильтры, настройки графика):
корень с переключаемым классом ``open``, ярлычок у кромки, выезжающее тело и
dcc.Store с состоянием. Отличия задаются параметрами:

- ``side``  — "left" или "right": с какого края экрана выезжает панель;
- ``mode``  — "reflow" (панель занимает место во flex-контейнере и раздвигает
  содержимое) или "overlay" (панель поверх содержимого между хэдером и футером);
- ``width`` — ширина открытой панели;
- ``tab_style`` — позиция ярлычка вдоль кромки (top/bottom/transform);
- ``content`` — список детей тела панели (или функция без аргументов).

Вся общая механика описана в assets/slide_panel.css; чтобы добавить новую
панель, достаточно создать экземпляр SlidePanel с нужным содержимым и вызвать
``render()``, а для переключения по ярлычку — ``register_toggle(app)``.
"""

import json

from dash import Input, Output, State, ctx, dcc, html
from dash_iconify import DashIconify


class SlidePanel:
    """Выезжающая панель: корень + ярлычок + тело + store состояния."""

    def __init__(
        self,
        root_id,
        tab_id,
        state_id,
        side="right",
        mode="reflow",
        width=299,
        tab_icon="tabler:settings",
        tab_label="",
        tab_title=None,
        tab_style=None,
        tab_extra_children=(),
        extra_tab_classes="",
        extra_root_classes=(),
        content=(),
        extra_stores=(),
    ):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if mode not in ("reflow", "overlay"):
            raise ValueError("mode must be 'reflow' or 'overlay'")
        self.root_id = root_id
        self.tab_id = tab_id
        self.state_id = state_id
        self.side = side
        self.mode = mode
        self.width = width
        self.tab_icon = tab_icon
        self.tab_label = tab_label
        self.tab_title = tab_title or tab_label
        self.tab_style = tab_style
        self.tab_extra_children = list(tab_extra_children)
        self.extra_tab_classes = extra_tab_classes
        self.extra_root_classes = list(extra_root_classes)
        self.content = content
        self.extra_stores = list(extra_stores)

    @property
    def closed_class(self):
        """className корня в закрытом состоянии."""
        return " ".join(
            [
                "slide-panel",
                "slide-panel--" + self.side,
                "slide-panel--" + self.mode,
            ]
            + self.extra_root_classes
        )

    @property
    def open_class(self):
        """className корня в открытом состоянии."""
        return self.closed_class + " open"

    @property
    def tab_class(self):
        """Базовый className ярлычка (для callback'ов, перезаписывающих класс)."""
        classes = "slide-panel__tab"
        if self.extra_tab_classes:
            classes += " " + self.extra_tab_classes
        return classes

    def render(self):
        tab_classes = self.tab_class
        icon = (
            DashIconify(icon=self.tab_icon, width=15)
            if isinstance(self.tab_icon, str)
            else self.tab_icon
        )
        tab = html.Div(
            [icon, html.Span(self.tab_label, className="slide-panel__tab-label")]
            + self.tab_extra_children,
            id=self.tab_id,
            className=tab_classes,
            style=self.tab_style,
            title=self.tab_title,
        )
        body = html.Div(self._content_children(), className="slide-panel__body")
        if self.mode == "overlay":
            # Ярлычок прикреплён к движущемуся блоку: в закрытом состоянии он
            # виден у кромки экрана, в открытом — у левого края панели.
            moving_parts = [html.Div([tab, body], className="slide-panel__mover")]
        else:
            moving_parts = [tab, body]
        return html.Div(
            moving_parts + [dcc.Store(id=self.state_id, data=False)] + self.extra_stores,
            id=self.root_id,
            className=self.closed_class,
            style={"--sp-width": f"{self.width}px"},
        )

    def _content_children(self):
        return list(self.content() if callable(self.content) else self.content)

    def register_toggle(self, app, open_inputs=(), close_inputs=()):
        """Переключение ярлычком; open_inputs всегда открывают, close_inputs закрывают."""
        open_ids = tuple(open_inputs)
        close_ids = tuple(close_inputs)

        @app.callback(
            Output(self.root_id, "className"),
            Output(self.state_id, "data"),
            Input(self.tab_id, "n_clicks"),
            *[Input(component_id, "n_clicks") for component_id in open_ids],
            *[Input(component_id, "n_clicks") for component_id in close_ids],
            State(self.state_id, "data"),
            prevent_initial_call=True,
        )
        def _toggle(_tab_clicks, *rest):
            opened = rest[-1]
            trigger = ctx.triggered_id
            if trigger in close_ids:
                should_open = False
            elif trigger in open_ids:
                should_open = True
            else:
                should_open = not bool(opened)
            return (self.open_class if should_open else self.closed_class), should_open

        return _toggle

    def register_outside_close(self, app, enabled_id, sink_id):
        """Close an open panel on an outside click when the checkbox is enabled."""
        abort_key = f"__slidePanelOutsideAbort_{self.root_id}"
        script = f"""
            function (enabled) {{
                var abortKey = {json.dumps(abort_key)};
                if (window[abortKey]) {{
                    window[abortKey].abort();
                    window[abortKey] = null;
                }}
                if (enabled) {{
                    var controller = new AbortController();
                    window[abortKey] = controller;
                    document.addEventListener("mousedown", function (event) {{
                        var panel = document.getElementById({json.dumps(self.root_id)});
                        if (!panel || !panel.classList.contains("open")) return;
                        if (panel.contains(event.target)) return;
                        window.dash_clientside.set_props(
                            {json.dumps(self.root_id)},
                            {{className: {json.dumps(self.closed_class)}}}
                        );
                        window.dash_clientside.set_props(
                            {json.dumps(self.state_id)},
                            {{data: false}}
                        );
                    }}, {{signal: controller.signal}});
                }}
                return Boolean(enabled);
            }}
        """
        return app.clientside_callback(
            script,
            Output(sink_id, "data"),
            Input(enabled_id, "checked"),
        )
