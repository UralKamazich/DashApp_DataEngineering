/**
 * Контекстное меню для графика (правый клик).
 * Перехватывает contextmenu до Plotly и показывает кастомное меню.
 */
(function () {
  "use strict";

  let _menuVisible = false;
  let _activeWorkspace = null;
  let _menuOrigin = {x: 0, y: 0};

  /* ---- Создание меню (один раз) ---- */
  function createMenu() {
    if (document.getElementById("graph-ctx-menu")) return;

    const menu = document.createElement("div");
    menu.id = "graph-ctx-menu";
    menu.setAttribute("role", "menu");
    menu.setAttribute("aria-label", "Действия с графиком");
    menu.innerHTML =
      '<div class="ctx-header">' +
        '<span class="ctx-header-title">График</span>' +
        '<span class="ctx-header-hint">Экспорт и настройки</span>' +
      '</div>' +
      '<div class="ctx-group" aria-label="Экспорт">' +
        menuItem("save-png", "image", "Сохранить PNG", "в файл", true) +
        menuItem("copy-png", "copy", "Копировать PNG", "в буфер") +
        menuItem("download-html", "code", "Сохранить HTML", "интерактивный") +
      '</div>' +
      '<div class="ctx-sep"></div>' +
      '<div class="ctx-group" aria-label="Вид">' +
        menuItem("reset-view", "reset", "Сбросить масштаб", "исходный вид") +
        menuItem("refresh", "refresh", "Обновить график") +
        menuItem("clear-graph", "eraser", "Очистить график", "без выгрузки") +
      '</div>' +
      '<div class="ctx-sep"></div>' +
      '<div class="ctx-group" aria-label="Оформление">' +
        menuItem("change-colors", "palette", "Сменить цвета") +
        menuItem("open-settings", "settings", "Общие настройки", "для всех типов") +
      '</div>';

    menu.addEventListener("click", handleItemClick);
    document.body.appendChild(menu);
  }

  function icon(name) {
    const paths = {
      image: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m3 16 5-5 4 4 3-3 6 6"/><circle cx="16" cy="9" r="1.5"/>',
      copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
      code: '<path d="m8 9-3 3 3 3M16 9l3 3-3 3M14 5l-4 14"/>',
      reset: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
      refresh: '<path d="M20 7h-5V2M4 17h5v5"/><path d="M5.1 9A8 8 0 0 1 18 5l2 2M18.9 15A8 8 0 0 1 6 19l-2-2"/>',
      eraser: '<path d="m7 21-4-4L14.5 5.5a2.1 2.1 0 0 1 3 0l1 1a2.1 2.1 0 0 1 0 3L7 21Z"/><path d="m11 9 4 4M7 21h13"/>',
      palette: '<path d="M12 3a9 9 0 0 0 0 18h1.5a2 2 0 0 0 0-4H12a2 2 0 0 1 0-4h5a4 4 0 0 0 4-4c0-3.3-4-6-9-6Z"/><circle cx="7.5" cy="10.5" r="1"/><circle cx="9.5" cy="6.5" r="1"/><circle cx="14.5" cy="6.5" r="1"/>',
      settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6 1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1Z"/>',
    };
    return '<span class="ctx-icon" aria-hidden="true"><svg viewBox="0 0 24 24">' + paths[name] + '</svg></span>';
  }

  function menuItem(action, iconName, label, note, primary) {
    return '<button class="ctx-item' + (primary ? ' ctx-item--primary' : '') + '" type="button" role="menuitem" data-action="' + action + '">' +
      icon(iconName) +
      '<span class="ctx-label">' + label + '</span>' +
      (note ? '<span class="ctx-note">' + note + '</span>' : '') +
    '</button>';
  }

  /* ---- Позиционирование меню ---- */
  function showMenu(x, y) {
    const menu = document.getElementById("graph-ctx-menu");
    if (!menu) return;

    _menuOrigin = {x: x, y: y};
    const settingsItem = menu.querySelector('[data-action="open-settings"]');
    if (settingsItem) {
      settingsItem.hidden = !(_activeWorkspace && _activeWorkspace.getAttribute("data-settings-popup-id"));
    }
    menu.style.display = "block";
    _menuVisible = true;

    const mw = menu.offsetWidth;
    const mh = menu.offsetHeight;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    menu.style.left = (x + mw > vw ? Math.max(0, vw - mw - 8) : x) + "px";
    menu.style.top = (y + mh > vh ? Math.max(0, vh - mh - 8) : y) + "px";
  }

  function hideMenu() {
    const menu = document.getElementById("graph-ctx-menu");
    if (menu) menu.style.display = "none";
    _menuVisible = false;
  }

  /* ---- Обработчик клика по пункту меню ---- */
  function handleItemClick(e) {
    const item = e.target.closest(".ctx-item");
    if (!item) return;

    const action = item.getAttribute("data-action");
    hideMenu();
    executeAction(action);
  }

  /* ---- Клик по действию активного GraphWorkspace ---- */
  function clickAction(action) {
    if (!_activeWorkspace) return;
    const id = _activeWorkspace.getAttribute("data-action-" + action);
    const btn = document.getElementById(id);
    if (btn) btn.click();
  }

  /* ---- Действия ---- */
  function executeAction(action) {
    switch (action) {
      case "refresh":
        clickAction("refresh");
        break;

      case "download-html":
        clickAction("download-html");
        break;

      case "copy-png":
        clickAction("copy-png");
        break;

      case "save-png":
        clickAction("save-png");
        break;

      case "change-colors":
        clickAction("change-colors");
        break;

      case "reset-view":
        resetGraphView();
        break;

      case "clear-graph":
        clickAction("clear-graph");
        break;

      case "open-settings":
        if (_activeWorkspace && window.graphSettingsPopover) {
          window.graphSettingsPopover.open(
            _activeWorkspace,
            "common",
            _menuOrigin.x,
            _menuOrigin.y
          );
        }
        break;
    }
  }

  /* ---- Сброс масштаба графика ---- */
  function resetGraphView() {
    const graphId = _activeWorkspace && _activeWorkspace.getAttribute("data-graph-id");
    const host = graphId && document.getElementById(graphId);
    if (!host) return;
    const gd = host.querySelector(".js-plotly-plot");
    if (!gd || !window.Plotly) return;

    try {
      // Use Plotly's own modebar actions first. They correctly reset all
      // cartesian/facet axes and understand 3D camera defaults.
      const nativeReset =
        gd.querySelector('.modebar-btn[data-attr="zoom"][data-val="reset"]') ||
        gd.querySelector('.modebar-btn[data-attr="resetDefault"]');

      if (nativeReset) {
        nativeReset.click();
        return;
      }

      // Fallback for custom Plotly configs where reset buttons are removed.
      const update = {};
      const layout = gd._fullLayout || gd.layout || {};

      Object.keys(layout).forEach(function (key) {
        if (/^[xy]axis\d*$/.test(key)) {
          update[key + ".autorange"] = true;
          update[key + ".range"] = null;
        }

        if (/^scene\d*$/.test(key)) {
          update[key + ".camera"] = null;
        }

        if (/^polar\d*$/.test(key)) {
          update[key + ".radialaxis.autorange"] = true;
          update[key + ".radialaxis.range"] = null;
        }
      });

      if (Object.keys(update).length > 0) {
        Plotly.relayout(gd, update);
      }
    } catch (err) {
      console.warn("[ctx-menu] reset error:", err);
    }
  }

  /* ---- Глобальные обработчики ---- */
  function setupGlobalListeners() {
    document.addEventListener("contextmenu", function (e) {
      // Drop zones own their right-click field picker.
      if (e.target.closest(".graph-drop-zone")) return;
      const graphHost = e.target.closest(".graph-workspace-plot");
      const workspace = graphHost && graphHost.closest(".graph-workspace");
      if (!workspace) return;

      e.preventDefault();
      e.stopPropagation();
      _activeWorkspace = workspace;
      showMenu(e.clientX, e.clientY);
    }, true);

    document.addEventListener("mousedown", function (e) {
      if (_menuVisible && !e.target.closest("#graph-ctx-menu")) {
        hideMenu();
      }
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && _menuVisible) {
        hideMenu();
      }
    });

    document.addEventListener("scroll", function () {
      if (_menuVisible) hideMenu();
    }, true);
  }

  /* ---- Инициализация ---- */
  function init() {
    createMenu();
    setupGlobalListeners();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
