# -*- coding: utf-8 -*-
"""
Скрипт запуска Dash-сервера для Electron.
"""

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
