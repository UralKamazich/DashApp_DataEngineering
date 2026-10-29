# -*- coding: utf-8 -*-
"""
Скрипт запуска Dash-сервера для Electron.
"""

import multiprocessing

# Required before importing numpy/sklearn/joblib in a frozen Windows process.
# Without it, worker processes can recursively relaunch the Electron server.
multiprocessing.freeze_support()

# Импорт app запускает layout и регистрацию callbacks
import app as application
from config import PORT

print(f"[INFO] Registered callbacks: {len(application.app.callback_map)}")
print(f"[INFO] Starting server on http://127.0.0.1:{PORT}")

application.app.run(
    host="127.0.0.1",
    port=PORT,
    debug=False,
    use_reloader=False,
)
