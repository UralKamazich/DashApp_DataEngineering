# -*- coding: utf-8 -*-
"""
Callbacks: переключение страниц и подсветка навигации.
"""

from dash import Output, Input, State, no_update
from dash_app import app
from components import dropdown_chart_type


MAIN_CHART_TYPES = {item["value"] for item in dropdown_chart_type.data}


def _normalize_main_chart_type(value):
    return value if value in MAIN_CHART_TYPES else "Scatter"


@app.callback(
    Output("segmented", "value"),
    Input("url", "pathname"),
    State("segmented", "value"),
)
def normalize_main_chart_type(_pathname, current_value):
    normalized = _normalize_main_chart_type(current_value)
    return no_update if normalized == current_value else normalized


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
