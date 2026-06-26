# -*- coding: utf-8 -*-
"""
Центральный файл приложения: создание Dash, layout и регистрация callbacks.
"""

import logging
from logging.handlers import RotatingFileHandler

import dash
from dash import Dash, _dash_renderer
_dash_renderer._set_react_version("18.2.0")

# =========================
# Логирование
# =========================
logger = logging.getLogger("dash-app")
NOISY_DEBUG = False
if not logger.handlers:
    logger.setLevel(logging.WARNING)  # Уменьшаем уровень логирования до WARNING, чтобы убрать info-сообщения
    _handler = RotatingFileHandler("dash_app.log", maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    _fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)

logging.basicConfig(level=logging.WARNING)  # Уменьшаем глобальное логирование
logger = logging.getLogger(__name__)

# =========================
# Приложение
# =========================
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="DataAnalize ver.1.0.23 DnD Y by Muslimov Ural",
)
server = app.server

# =========================
# Layout
# =========================
from layout import create_layout
app.layout = create_layout()

# =========================
# Регистрация callbacks (импорт автоматически регистрирует @app.callback)
# =========================
from callbacks import modals, file_handling, filters, pipeline, data_engineering, dropdowns, graph, colors, download, columns_sidebar

# =========================
# Shutdown route
# =========================
@app.server.route("/_shutdown", methods=["POST"])
def _shutdown():
    from utils import _shutdown_server
    _shutdown_server()
    return "Server shutting down..."