# -*- coding: utf-8 -*-
"""
Callbacks: открытие/закрытие drawer и переключение страниц + подсветка навигации.
"""

from dash import Output, Input, State, no_update
from dash.exceptions import PreventUpdate
from dash_app import app


# ============ Переключение страниц + подсветка активной ссылки (клиентский колбэк) ============
app.clientside_callback(
    """
    function(pathname) {
        // --- Показать только нужную страницу ---
        var pages = {
            "/": "page-graph",
            "/correlation": "page-correlation",
            "/data-engineering": "page-data-engineering",
            "/clustering": "page-clustering",
            "/ml": "page-ml"
        };
        for (var key in pages) {
            var el = document.getElementById(pages[key]);
            if (el) el.style.display = "none";
        }
        var targetId = pages[pathname] || pages["/"];
        var target = document.getElementById(targetId);
        if (target) target.style.display = "block";

        // --- Подсветить активную ссылку ---
        var links = document.querySelectorAll('.nav-link');
        for (var i = 0; i < links.length; i++) {
            var link = links[i];
            if (link.getAttribute('href') === pathname) {
                link.style.color = '#fff';
                link.style.backgroundColor = 'rgba(255,255,255,0.12)';
            } else {
                link.style.color = '#aaa';
                link.style.backgroundColor = 'transparent';
            }
        }

        return pathname;
    }
    """,
    Output("nav-active-store", "data"),
    Input("url", "pathname"),
)


# ============ Диалог настройки (открыть дравер) ============
@app.callback(
    Output("drawer-simple", "opened"),
    Input("context-menu-btn", "n_clicks"),
    prevent_initial_call=True,
)
def drawer_demo(n_clicks):
    return True


# ============ Drag-and-Drop: плашки → дропдауны осей ============
app.clientside_callback(
    """
    function() {
        if (window.__dndInstalled) return window.dash_clientside.no_update;
        window.__dndInstalled = true;

        var isDragging = false;
        var isGraphHover = false;

        function getGraphContainer() {
            var graph = document.getElementById('graph');
            if (!graph) return null;
            var c = graph.parentElement;
            if (c) c.classList.add('graph-container');
            return c;
        }

        function getAllZones() { return document.querySelectorAll('.graph-drop-zone'); }

        function updateZonesVisibility() {
            var show = isGraphHover || isDragging;
            getAllZones().forEach(function(z) {
                z.style.display = show ? 'flex' : 'none';
            });
            var c = getGraphContainer();
            if (c) {
                if (isDragging) {
                    c.classList.add('drag-active');
                    c.classList.remove('graph-hover');
                } else if (isGraphHover) {
                    c.classList.remove('drag-active');
                    c.classList.add('graph-hover');
                } else {
                    c.classList.remove('drag-active');
                    c.classList.remove('graph-hover');
                }
            }
        }

        // --- Hover: показывать зоны при наведении на график ---
        var container = getGraphContainer();
        if (container) {
            container.addEventListener('mouseenter', function() {
                isGraphHover = true;
                updateZonesVisibility();
            });
            container.addEventListener('mouseleave', function() {
                isGraphHover = false;
                updateZonesVisibility();
            });
        }

        // --- Drag start ---
        document.addEventListener('dragstart', function(e) {
            var badge = e.target.closest('[data-column-name]');
            if (!badge) return;
            var colName = badge.getAttribute('data-column-name');
            if (!colName) return;
            e.dataTransfer.setData('text/plain', colName);
            e.dataTransfer.effectAllowed = 'move';
            badge.style.opacity = '0.4';
            isDragging = true;
            updateZonesVisibility();
        });

        // --- Drag end ---
        document.addEventListener('dragend', function(e) {
            var badge = e.target.closest('[data-column-name]');
            if (badge) badge.style.opacity = '1';
            isDragging = false;
            document.querySelectorAll('.zone-hover').forEach(function(z) { z.classList.remove('zone-hover'); });
            updateZonesVisibility();
        });

        // --- Drag over (разрешить drop) ---
        document.addEventListener('dragover', function(e) {
            var zone = e.target.closest('.graph-drop-zone');
            if (zone) {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
            }
        });

        // --- Подсветка зон: dragenter/dragleave напрямую на зонах ---
        getAllZones().forEach(function(zone) {
            zone.addEventListener('dragenter', function(e) {
                e.preventDefault();
                zone.classList.add('zone-hover');
            });
            zone.addEventListener('dragleave', function() {
                zone.classList.remove('zone-hover');
            });
            zone.addEventListener('dragover', function(e) {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                zone.classList.add('zone-hover');
            });
            // ПКМ — очистить дроп-зону
            zone.addEventListener('contextmenu', function(e) {
                e.preventDefault();
                e.stopPropagation();
                var targetId = zone.getAttribute('data-drop-target');
                if (targetId) {
                    dash_clientside.set_props(targetId, {value: null});
                }
            });
        });

        // --- Drop ---
        document.addEventListener('drop', function(e) {
            var zone = e.target.closest('.graph-drop-zone');
            if (!zone) return;
            e.preventDefault(); e.stopPropagation();
            var colName = e.dataTransfer.getData('text/plain');
            var targetId = zone.getAttribute('data-drop-target');
            isDragging = false;
            zone.classList.remove('zone-hover');
            updateZonesVisibility();
            if (colName && targetId) {
                dash_clientside.set_props(targetId, {value: colName});
            }
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output("nav-active-store", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call='initial_duplicate',
)


# ============ Обновление подписей дроп-зон при изменении dropdown ============
app.clientside_callback(
    """
    function(xVal, yVal, colorVal, sizeVal, textVal) {
        function updateLabel(spanId, defaultText, value) {
            var el = document.getElementById(spanId);
            if (el) el.textContent = value || defaultText;
        }
        updateLabel('zone-label-x', 'X', xVal);
        updateLabel('zone-label-y', 'Y', yVal);
        updateLabel('zone-label-color', 'Color', colorVal);
        updateLabel('zone-label-size', 'Size', sizeVal);
        updateLabel('zone-label-text', 'Подпись', textVal);
        return window.dash_clientside.no_update;
    }
    """,
    Output("nav-active-store", "data", allow_duplicate=True),
    Input("dropdown_x", "value"),
    Input("dropdown_y", "value"),
    Input("dropdown_color", "value"),
    Input("dropdown_size", "value"),
    Input("dropdown_text", "value"),
    prevent_initial_call='initial_duplicate',
)
