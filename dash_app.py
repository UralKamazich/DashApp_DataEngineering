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


# Optional per-callback profiling: DATAANALIZE_PROFILE=1 logs every
# _dash-update-component request with its duration and response size.
import os as _os

if _os.environ.get("DATAANALIZE_PROFILE") == "1":
    import time as _time
    from flask import g as _g, request as _request

    @server.before_request
    def _profile_start():
        _g._profile_t0 = _time.perf_counter()

    @server.after_request
    def _profile_end(response):
        if _request.path.endswith("_dash-update-component"):
            elapsed = _time.perf_counter() - getattr(_g, "_profile_t0", _time.perf_counter())
            try:
                body = _request.get_json(silent=True) or {}
                output = str(body.get("output", "?"))[:110]
                req_kb = (_request.content_length or 0) / 1024
            except Exception:
                output, req_kb = "?", 0
            resp_kb = (response.calculate_content_length() or 0) / 1024
            print(
                f"[PROFILE] {elapsed*1000:8.0f} ms  req {req_kb:8.0f} KB  "
                f"resp {resp_kb:8.0f} KB  {output}",
                flush=True,
            )
        return response
