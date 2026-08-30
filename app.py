# -*- coding: utf-8 -*-
"""
Центральный файл приложения: layout и регистрация callbacks.
"""

import logging
from logging.handlers import RotatingFileHandler

# =========================
# Логирование
# =========================
logger = logging.getLogger("dash-app")
if not logger.handlers:
    logger.setLevel(logging.WARNING)
    _handler = RotatingFileHandler("dash_app.log", maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    _fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)

logging.basicConfig(level=logging.WARNING)

# =========================
# Импорт экземпляра приложения
# =========================
from dash_app import app

# =========================
# Layout
# =========================
from layout import (
    CORRELATION_WORKSPACE,
    DASHBOARD_WORKSPACES,
    GRAPH_WORKSPACE,
    MULTI_Y_WORKSPACE,
    MULTIVARIATE_WORKSPACE,
    create_layout,
)
app.layout = create_layout()
GRAPH_WORKSPACE.register_callbacks(app)
for dashboard_workspace in DASHBOARD_WORKSPACES:
    dashboard_workspace.register_callbacks(app)
MULTI_Y_WORKSPACE.register_callbacks(app)
CORRELATION_WORKSPACE.register_callbacks(app)
MULTIVARIATE_WORKSPACE.register_callbacks(app)

# =========================
# Регистрация callbacks
# =========================
from callbacks import (
    clustering,
    columns_sidebar,
    data_engineering,
    dashboard,
    dataset_panel,
    datasets,
    download,
    dropdowns,
    file_handling,
    filters,
    graph,
    ml,
    ml_neural_networks,
    ml_random_forest,
    modals,
    multivariate,
    pipeline,
)
