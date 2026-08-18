# DataAnalize

Dash-приложение для визуализации и обработки локальных Excel/PKL-датасетов.

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
