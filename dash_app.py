# -*- coding: utf-8 -*-
"""
Экземпляр Dash-приложения. Вынесен отдельно для избежания циклических импортов.
"""

import dash
from dash import Dash, _dash_renderer
from config import APP_TITLE
_dash_renderer._set_react_version("18.2.0")

# Кастомный index_string для добавления глобальных стилей
CUSTOM_INDEX_STRING = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0 !important;
                padding: 0 !important;
                height: 100vh !important;
                overflow: hidden !important;
            }
            #react-entry-point,
            #_dash-app-content,
            .dash-renderer {
                height: 100% !important;
                overflow: hidden !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title=APP_TITLE,
    index_string=CUSTOM_INDEX_STRING,
)
server = app.server
