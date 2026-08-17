# -*- coding: utf-8 -*-
"""
Экземпляр Dash-приложения. Вынесен отдельно для избежания циклических импортов.
"""

import dash
from dash import Dash, _dash_renderer
_dash_renderer._set_react_version("18.2.0")

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="DataAnalize ver.1.0.27 collapsed panel by Muslimov Ural",
)
server = app.server
