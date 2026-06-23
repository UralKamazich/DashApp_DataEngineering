# -*- coding: utf-8 -*-
"""
Точка входа: запуск Dash-приложения с поиском свободного порта и автоматическим открытием браузера.
"""

if __name__ == "__main__":
    import threading
    import time
    import socket
    import webbrowser
    import contextlib
    import sys
    import os

    from app import app

    def get_lan_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    def pick_free_port(start_port: int) -> int:
        port = start_port
        while True:
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("", port))
                    return port
                except OSError:
                    port += 1

    PORT = 8090
    PORT = pick_free_port(PORT)
    LAN_IP = get_lan_ip()

    IS_FROZEN = getattr(sys, "frozen", False)
    FORCE_LOCALHOST = os.environ.get("OPEN_LOCALHOST", "1") == "1"

    def wait_and_open():
        url_local = f"http://127.0.0.1:{PORT}"
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", PORT), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.1)

        if FORCE_LOCALHOST or IS_FROZEN:
            webbrowser.open(url_local)
        else:
            webbrowser.open(url_local)

        print("\n================ DASH APP =================")
        print(f" Локально (буфер ОК): http://127.0.0.1:{PORT}")
        print(f" В сети LAN (без буфера): http://{LAN_IP}:{PORT}")
        print("==========================================\n")

    threading.Thread(target=wait_and_open, daemon=True).start()

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False, threaded=True)