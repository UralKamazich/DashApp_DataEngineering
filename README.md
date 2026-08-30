# DataAnalize

Dash-приложение для визуализации, обработки и ML-анализа табличных datasets.

Поддерживаемые источники: локальные Excel, CSV, TXT, TSV, ZIP и PKL; прямые
CSV/TXT/TSV/ZIP-ссылки; встроенный каталог популярных учебных datasets. ZIP с
одной таблицей открывается автоматически, а для нескольких таблиц приложение
показывает выбор. Для
текстовых таблиц автоматически определяются кодировка, разделитель и десятичный
знак. Интернет-источник после загрузки работает как обычный исходный dataset во
всех разделах приложения.

## Окружение

Проверенная версия Python — 3.14.6.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Запуск

В браузере:

```bash
.venv/bin/python run.py
```

Electron-обёртка в режиме разработки:

```bash
npm ci
npm start
```

В Windows полный debug-режим подготавливается и запускается одной командой из
PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dev_windows.ps1
```

Скрипт создаёт `.venv` через Python 3.14, ставит Python/Node-зависимости и
запускает Electron. При следующих запусках установку можно пропустить:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/dev_windows.ps1 -SkipInstall
```

Electron автоматически использует `.venv\\Scripts\\python.exe`. Кнопка выбора
файла открывает нативный диалог Electron, одинаковый по возможностям с
macOS-версией.

## Windows-сборка

Workflow `.github/workflows/build-windows.yml` запускается вручную через
GitHub Actions (`Run workflow`), когда потребуется готовая сборка. Windows-runner:

1. устанавливает все Python-зависимости и выполняет полный набор тестов;
2. собирает самостоятельный `dataanalize-server.exe` через PyInstaller;
3. упаковывает полный Electron-интерфейс в NSIS installer и ZIP;
4. публикует результаты как artifact `DataAnalize-2.0.0-Windows-x64`.

На компьютере пользователя отдельная установка Python не требуется.

## Проверка

Smoke-тесты используют стандартный модуль `unittest`, поэтому отдельная тестовая
зависимость не требуется:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## GraphWorkspace

Визуальная оболочка основного Plotly-графика вынесена в `graph_workspace.py`.
Она размещает управляющую панель и drag-and-drop цели рядом с `dcc.Graph`, а не
внутри него. Поэтому PNG/HTML-экспорт содержит только Plotly figure. Вычисление
figure и callback-логика остаются независимыми от оболочки.

## MultiYAxisWorkspace

Постоянная страница `/multi-y` использует отдельный класс из
`multi_axis_workspace.py`; исходная страница «График» при этом остаётся
независимой. Один экземпляр хранит собственные dataset, слой данных, общий X,
серии и оси. Числовой канал можно бросить на левую или правую DnD-зону, после
чего его серия, цвет, сторона и шкала настраиваются отдельно. Чистая фигура
строится движком `multi_axis_engine.py`, поэтому тот же компонент можно
создавать повторно в будущих dashboard-layouts с другим `graph_id`.
