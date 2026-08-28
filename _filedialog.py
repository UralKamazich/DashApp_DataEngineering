# -*- coding: utf-8 -*-
"""Вспомогательный скрипт: открывает диалог выбора файла в главном потоке отдельного процесса."""

import sys
import os

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    path = filedialog.askopenfilename(
        title="Выберите файл данных",
        filetypes=[
            ("Excel files", "*.xlsx"),
            ("CSV files", "*.csv"),
            ("Text tables", "*.txt *.tsv"),
            ("ZIP archives", "*.zip"),
            ("Pickle files", "*.pkl"),
            ("Supported datasets", "*.xlsx *.csv *.txt *.tsv *.zip *.pkl"),
            ("All files", "*.*"),
        ],
    )
    try:
        root.destroy()
    except Exception:
        pass

    if path:
        print(path)
