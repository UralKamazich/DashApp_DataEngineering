# -*- coding: utf-8 -*-
"""
Callbacks: переключение страниц и подсветка навигации.
"""

from dash import Output, Input
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
