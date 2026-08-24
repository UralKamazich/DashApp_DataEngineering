# Конспект проекта DataAnalize

## Состояние на 24.08.2026

### Структура проекта
```
DashApp_DataEngineering/
├── run.py              # Точка входа: запуск на свободном порту, автооткрытие браузера
├── app.py              # Dash-приложение, layout, регистрация callbacks
├── config.py           # Константы, стили Plotly, легенда, COLOR_THRESHOLD, MAX_FILTERS
├── utils.py            # Вспомогательные функции (чтение Store, пустая фигура, уведомления и т.д.)
├── components.py       # UI-компоненты (dropdowns, selects, switches, кнопки)
├── layout.py           # Layout приложения (многостраничный)
├── _filedialog.py      # Отдельный процесс tkinter для macOS (выбор файла)
├── callbacks/
│   ├── __init__.py
│   ├── modals.py        # Клиентский колбэк переключения страниц + подсветка nav + открытие Drawer
│   ├── file_handling.py # Загрузка .xlsx/.pkl, выбор листа, инфо о файле
│   ├── filters.py       # Управление фильтрами (добавление, удаление, обновление)
│   ├── pipeline.py      # Применение фильтров к активному dataset
│   ├── clustering.py    # Расчёт, визуализация и запись кластеризации
│   ├── data_engineering.py # Текстовые копии столбцов
│   ├── dropdowns.py     # Обновление опций всех дропдаунов при смене данных
│   ├── graph.py         # Основной график (update_main_graph) + нижние (update_lower_graphs)
│   ├── colors.py        # Диалог выбора цветов
│   └── download.py      # Скачивание HTML/Excel/PNG
└── .venv/              # Python 3.14, виртуальное окружение
```

### Архитектура страниц (многостраничное приложение)
Навигация — ссылки в тёмном хедере. Клиентский колбэк (`modals.py`) скрывает/показывает div-ы через `display: none/block`.

| URL | Страница | Левая панель | Правая панель |
|-----|----------|-------------|---------------|
| `/` | График | Оси X/Y/Z, Color, Size, Text, Facet, Hover, Фильтры | Тип графика + кнопки + график |
| `/correlation` | Коррелограмма | Выбор корреляционных столбцов | Корреляционная матрица + bar-графики |
| `/data-engineering` | Data Engineering | Биннинг, текстовые копии, агрегаты | График |
| `/clustering` | Кластеризация | Dataset-aware лаборатория | PCA, подбор K, размеры и профили |
| `/ml/experiments` | ML · Эксперименты | Журнал запусков | Сравнение качества |
| `/ml/catboost` | ML · CatBoost | Регрессия/классификация, валидация и параметры | Метрики, графики, SHAP, диагностика |
| `/ml/random-forest` | ML · Random Forest | Регрессия/классификация, split/CV/group/time | Метрики, OOB, важность, диагностика |
| `/ml/neural-networks` | ML · Нейросети | Подлист подготовлен | Модель ещё не подключена |

### Что сделано за эту сессию
1. **Перестройка UI**: тёмный хедер (`#1A1B1E`) с названием, кнопкой выбора файла и навигацией
2. **SegmentedControl → Dropdown**: типы графиков теперь в `dmc.Select` (id="segmented" сохранён)
3. **Многостраничность**: 5 страниц, переключение через `dcc.Location` + клиентский JS колбэк
4. **Подсветка активной страницы**: белый текст + серый фон на активной ссылке
5. **Информация о файле**: внизу под графиком серым шрифтом (имя, лист, строки, столбцы, путь)
6. **Убрана DE-модалка**: функционал Data Engineering разнесён на отдельные страницы

### Что нужно доделать
1. ~~**Информация о файле появляется только после обновления графика**~~ ✅ Исправлено (24.06.2026): file-info-bar перенесён в левый нижний угол, source-file-path добавлен как Input, зелёное status-message убрано
2. **Страница ML** — CatBoost и Random Forest поддерживают регрессию/классификацию, фоновые задания, прогресс/отмену, отдельные результаты и запись прогнозов в производный dataset; далее — нейросети
3. **Фильтры на странице Корреляции** — сейчас их нет, добавить
4. **DE-страница**: добавить фильтры для биннинга/агрегатов
5. **Страница кластеризации**: расчёт отделён от записи; результат добавляется в текущий или новый dataset

### Как запустить
```bash
cd /Users/uralmuslimov/Desktop/CodeDir2/DashApp_DataEngineering
source .venv/bin/activate
python run.py
# Откроется http://127.0.0.1:8090
```

### Важные ID компонентов
- `segmented` — тип графика (dmc.Select)
- `pick-file-btn` — кнопка выбора файла
- `file-info-bar` — строка информации о файле внизу
- `page-graph`, `page-correlation`, `page-data-engineering`, `page-clustering`, `page-ml` — контейнеры страниц
- `ml-page-experiments`, `ml-page-catboost`, `ml-page-random-forest`, `ml-page-neural-networks` — подлисты ML
- `url` — dcc.Location
- `nav-active-store` — dcc.Store с текущим путём
- Основные совместимые ID сохранены (dropdown_x, dropdown_y, update-graf, graph, btn-grouping, btn-agg, btn-txtcopy и т.д.)

### Колбэки, требующие осторожности
- `update_main_graph` (graph.py) — ~300 строк, все типы графиков
- `callbacks.clustering` — отдельный расчёт и явная запись кластерных каналов в реестр dataset
- `callbacks.ml` — CatBoost, фоновые задания, история экспериментов, экспорт Excel/CBM и стратегии разбиения
- `callbacks.ml_random_forest` — отдельное состояние Random Forest, OOB, важность признаков, экспорт Excel/joblib и запись результата
- `random_forest_engine.py` — sklearn Pipeline для смешанных данных, split/KFold/GroupKFold/TimeSeriesSplit, регрессия и классификация
- `ml_jobs.py` — однопоточная очередь тяжёлых ML-расчётов, polling прогресса и отмена
- `ml_tuning.py` — детерминированный CatBoost random search на validation holdout без промежуточных dataset
- `update_dropdown_options_all` (dropdowns.py) — 15 Outputs
- `create_text_copies` (data_engineering.py) — текстовые копии
