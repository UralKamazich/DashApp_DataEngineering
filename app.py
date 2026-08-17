# -*- coding: utf-8 -*-
"""
Центральный файл приложения: создание Dash, layout и регистрация callbacks.
"""

import logging
from logging.handlers import RotatingFileHandler

# =========================
# Логирование
# =========================
logger = logging.getLogger("dash-app")
NOISY_DEBUG = False
if not logger.handlers:
    logger.setLevel(logging.WARNING)
    _handler = RotatingFileHandler("dash_app.log", maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    _fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# =========================
# Импорт экземпляра приложения (из отдельного файла для избежания циклических импортов)
# =========================
from dash_app import app, server

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


# =========================
# Запуск
# =========================
if __name__ == "__main__":
    from pathlib import Path
    from config import PORT

    # SSL configuration
    ssl_context = None
    protocol = "http"
    cert_path = Path(__file__).parent / "cert.pem"
    key_path = Path(__file__).parent / "key.pem"

    if cert_path.exists() and key_path.exists():
        ssl_context = (str(cert_path), str(key_path))
        protocol = "https"
        print(f"✓ Running in HTTPS mode (port {PORT})")
    else:
        print("⚠ SSL certificates not found. Running in HTTP mode.")
        print("  Run 'python generate_cert.py' to generate certs.")

    print(f"[INFO] Registered callbacks: {len(app.callback_map)}")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        use_reloader=False,
        ssl_context=ssl_context,
    )
