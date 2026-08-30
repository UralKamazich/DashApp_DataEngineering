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
        "/dashboard": "page-dashboard",
            "/multi-y": "page-multi-y",
            "/correlation": "page-correlation",
            "/data-engineering": "page-data-engineering",
            "/clustering": "page-clustering",
            "/ml": "page-ml"
        };
        for (var key in pages) {
            var el = document.getElementById(pages[key]);
            if (el) el.style.display = "none";
        }
        var isMl = pathname === "/ml" || pathname.indexOf("/ml/") === 0;
        var targetId = isMl ? pages["/ml"] : (pages[pathname] || pages["/"]);
        var target = document.getElementById(targetId);
        if (target) target.style.display = "block";

        // --- Внутренняя маршрутизация ML: тяжёлая модель остаётся отдельным подлистом ---
        var mlPages = {
            "/ml": "ml-page-experiments",
            "/ml/data-profile": "ml-page-data-profile",
            "/ml/experiments": "ml-page-experiments",
            "/ml/catboost": "ml-page-catboost",
            "/ml/random-forest": "ml-page-random-forest",
            "/ml/neural-networks": "ml-page-neural-networks"
        };
        Object.keys(mlPages).forEach(function(key) {
            var page = document.getElementById(mlPages[key]);
            if (page) page.style.display = "none";
        });
        var mlTargetId = mlPages[pathname] || "ml-page-experiments";
        var mlTarget = document.getElementById(mlTargetId);
        if (isMl && mlTarget) mlTarget.style.display = "block";

        // --- Подсветить активную ссылку ---
        var links = document.querySelectorAll('.nav-link');
        for (var i = 0; i < links.length; i++) {
            var link = links[i];
            var href = link.getAttribute('href');
            if (href === pathname || (href === "/ml/experiments" && isMl)) {
                link.style.color = '#fff';
                link.style.backgroundColor = 'rgba(255,255,255,0.12)';
            } else {
                link.style.color = '#aaa';
                link.style.backgroundColor = 'transparent';
            }
        }

        var mlLinks = document.querySelectorAll('.ml-subnav-link');
        for (var j = 0; j < mlLinks.length; j++) {
            var mlLink = mlLinks[j];
            var active = mlLink.getAttribute('href') === pathname ||
                (pathname === "/ml" && mlLink.getAttribute('href') === "/ml/experiments");
            mlLink.classList.toggle('is-active', active);
        }

        return pathname;
    }
    """,
    Output("nav-active-store", "data"),
    Input("url", "pathname"),
)
