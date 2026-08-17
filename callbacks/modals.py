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

        function getAllZones() { return document.querySelectorAll('[data-drop-target]'); }
        function setZonesStyle(prop, value) { getAllZones().forEach(function(z) { z.style[prop] = value; }); }

        document.addEventListener('dragstart', function(e) {
            var badge = e.target.closest('[data-column-name]');
            if (!badge) return;
            var colName = badge.getAttribute('data-column-name');
            if (!colName) return;
            e.dataTransfer.setData('text/plain', colName);
            e.dataTransfer.effectAllowed = 'move';
            badge.style.opacity = '0.4';
            setZonesStyle('borderColor', '#2196F3');
            setZonesStyle('backgroundColor', 'rgba(33,150,243,0.08)');
            var overlay = document.getElementById('graph-drop-overlay');
            if (overlay) overlay.style.display = 'block';
        });

        document.addEventListener('dragend', function(e) {
            var badge = e.target.closest('[data-column-name]');
            if (badge) badge.style.opacity = '1';
            setZonesStyle('borderColor', 'transparent');
            setZonesStyle('backgroundColor', 'transparent');
            var overlay = document.getElementById('graph-drop-overlay');
            if (overlay) overlay.style.display = 'none';
        });

        document.addEventListener('dragover', function(e) {
            var zone = e.target.closest('[data-drop-target]');
            if (!zone) return;
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            zone.style.backgroundColor = 'rgba(33,150,243,0.2)';
        });

        document.addEventListener('dragleave', function(e) {
            var zone = e.target.closest('[data-drop-target]');
            if (!zone) return;
            zone.style.backgroundColor = 'rgba(33,150,243,0.1)';
        });

        document.addEventListener('drop', function(e) {
            var zone = e.target.closest('[data-drop-target]');
            if (!zone) return;
            e.preventDefault(); e.stopPropagation();
            var colName = e.dataTransfer.getData('text/plain');
            var targetId = zone.getAttribute('data-drop-target');
            setZonesStyle('borderColor', 'transparent');
            setZonesStyle('backgroundColor', 'transparent');
            var overlay = document.getElementById('graph-drop-overlay');
            if (overlay) overlay.style.display = 'none';
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
