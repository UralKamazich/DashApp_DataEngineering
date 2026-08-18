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


# ============ Размер Paper синхронизирован с настройками размера графика ============
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
    Output("graph-paper", "style"),
    Input("InputSizePlot", "value"),
    Input("InputSizePlotW", "value"),
)


# ============ Drag-and-Drop: плашки → поля GraphWorkspace ============
app.clientside_callback(
    """
    function() {
        if (window.__graphDndInstalled) return window.dash_clientside.no_update;
        var workspace = document.getElementById('graph-workspace');
        if (!workspace) return window.dash_clientside.no_update;
        window.__graphDndInstalled = true;

        var draggedBadge = null;

        function setDragging(active) {
            workspace.classList.toggle('dnd-active', active);
            if (!active) {
                workspace.querySelectorAll('.zone-hover').forEach(function(zone) {
                    zone.classList.remove('zone-hover');
                });
            }
        }

        function getZoneValue(zone) {
            try {
                return JSON.parse(zone.getAttribute('data-current-value') || 'null');
            } catch (error) {
                return null;
            }
        }

        function setFieldValue(zone, columnName) {
            var targetId = zone.getAttribute('data-drop-target');
            var mode = zone.getAttribute('data-drop-mode') || 'replace';
            if (!targetId || !columnName) return;

            var current = getZoneValue(zone);
            if (mode === 'append') {
                var values = Array.isArray(current) ? current.slice() : [];
                var index = values.indexOf(columnName);
                if (index === -1) values.push(columnName);
                else values.splice(index, 1);
                dash_clientside.set_props(targetId, {value: values});
                return;
            }

            // Повторный drop того же столбца очищает одиночное поле.
            dash_clientside.set_props(targetId, {
                value: current === columnName ? null : columnName
            });
        }

        function makeDragPreview(columnName) {
            var preview = document.createElement('div');
            preview.className = 'column-drag-preview';
            preview.textContent = columnName;
            document.body.appendChild(preview);
            return preview;
        }

        // --- Drag start ---
        document.addEventListener('dragstart', function(e) {
            var badge = e.target.closest('[data-column-name]');
            if (!badge) return;
            var colName = badge.getAttribute('data-column-name');
            if (!colName) return;

            draggedBadge = badge;
            e.dataTransfer.setData('text/plain', colName);
            e.dataTransfer.effectAllowed = 'copy';

            var preview = makeDragPreview(colName);
            e.dataTransfer.setDragImage(preview, 16, 16);
            window.setTimeout(function() { preview.remove(); }, 0);

            badge.classList.add('column-badge--dragging');
            setDragging(true);
        });

        // --- Drag end ---
        document.addEventListener('dragend', function() {
            if (draggedBadge) draggedBadge.classList.remove('column-badge--dragging');
            draggedBadge = null;
            setDragging(false);
        });

        // --- Drag over: делегирование работает и после обновления layout ---
        document.addEventListener('dragover', function(e) {
            var zone = e.target.closest('.graph-drop-zone');
            if (!zone || !workspace.contains(zone)) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            workspace.querySelectorAll('.zone-hover').forEach(function(item) {
                if (item !== zone) item.classList.remove('zone-hover');
            });
            zone.classList.add('zone-hover');
        });

        document.addEventListener('dragleave', function(e) {
            var zone = e.target.closest('.graph-drop-zone');
            if (!zone || !workspace.contains(zone)) return;
            if (!e.relatedTarget || !zone.contains(e.relatedTarget)) {
                zone.classList.remove('zone-hover');
            }
        });

        // --- Drop ---
        document.addEventListener('drop', function(e) {
            var zone = e.target.closest('.graph-drop-zone');
            if (!zone || !workspace.contains(zone)) return;
            e.preventDefault();
            e.stopPropagation();
            var colName = e.dataTransfer.getData('text/plain');
            setFieldValue(zone, colName);
            zone.classList.remove('zone-hover');
            if (draggedBadge) draggedBadge.classList.remove('column-badge--dragging');
            draggedBadge = null;
            setDragging(false);
        });

        // ПКМ по зоне во время drag очищает соответствующее поле.
        document.addEventListener('contextmenu', function(e) {
            var zone = e.target.closest('.graph-drop-zone');
            if (!zone || !workspace.contains(zone)) return;
            e.preventDefault();
            e.stopPropagation();
            var targetId = zone.getAttribute('data-drop-target');
            var emptyValue = zone.getAttribute('data-drop-mode') === 'append' ? [] : null;
            if (targetId) dash_clientside.set_props(targetId, {value: emptyValue});
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
    function(xVal, yVal, zVal, colorVal, sizeVal, textVal, facetRowVal, facetColVal, hoverVal) {
        var values = {
            dropdown_x: xVal,
            dropdown_y: yVal,
            dropdown_z: zVal,
            dropdown_color: colorVal,
            dropdown_size: sizeVal,
            dropdown_text: textVal,
            dropdown_facet_row: facetRowVal,
            dropdown_facet_col: facetColVal,
            dropdown_hover_data: hoverVal
        };

        document.querySelectorAll('.graph-drop-zone').forEach(function(zone) {
            var value = values[zone.getAttribute('data-drop-target')];
            var hasValue = Array.isArray(value) ? value.length > 0 : Boolean(value);
            var valueElement = zone.querySelector('.graph-drop-zone-value');
            if (valueElement) {
                valueElement.textContent = Array.isArray(value)
                    ? (value.join(', ') || 'Не выбрано')
                    : (value || 'Не выбрано');
            }
            zone.setAttribute('data-current-value', JSON.stringify(value ?? null));
            zone.classList.toggle('has-value', hasValue);
        });
        return window.dash_clientside.no_update;
    }
    """,
    Output("nav-active-store", "data", allow_duplicate=True),
    Input("dropdown_x", "value"),
    Input("dropdown_y", "value"),
    Input("dropdown_z", "value"),
    Input("dropdown_color", "value"),
    Input("dropdown_size", "value"),
    Input("dropdown_text", "value"),
    Input("dropdown_facet_row", "value"),
    Input("dropdown_facet_col", "value"),
    Input("dropdown_hover_data", "value"),
    prevent_initial_call='initial_duplicate',
)
