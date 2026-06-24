# -*- coding: utf-8 -*-
"""
Callbacks: открытие/закрытие drawer и переключение страниц + подсветка навигации.
"""

from dash import callback, Output, Input, State, no_update
from dash.exceptions import PreventUpdate
from app import app


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
            if (link.getAttribute('data-href') === pathname) {
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
@callback(
    Output("drawer-simple", "opened"),
    Input("drawer-demo-button", "n_clicks"),
    prevent_initial_call=True
)
def drawer_demo(n_clicks):
    return True