/**
 * Контекстное меню для графика (правый клик).
 * Перехватывает contextmenu до Plotly и показывает кастомное меню.
 */
(function () {
  "use strict";

  let _menuVisible = false;

  /* ---- Создание меню (один раз) ---- */
  function createMenu() {
    if (document.getElementById("graph-ctx-menu")) return;

    const menu = document.createElement("div");
    menu.id = "graph-ctx-menu";
    menu.innerHTML =
      '<div class="ctx-item" data-action="refresh">' +
        '<span class="ctx-icon">\u{1F504}</span>Обновить график' +
      '</div>' +
      '<div class="ctx-sep"></div>' +
      '<div class="ctx-item" data-action="download-html">' +
        '<span class="ctx-icon">\u{1F4C4}</span>Сохранить HTML' +
      '</div>' +
      '<div class="ctx-item" data-action="download-excel">' +
        '<span class="ctx-icon">\u{1F4CA}</span>Сохранить Excel' +
      '</div>' +
      '<div class="ctx-item" data-action="copy-png">' +
        '<span class="ctx-icon">\u{1F5BC}</span>Копировать PNG' +
      '</div>' +
      '<div class="ctx-sep"></div>' +
      '<div class="ctx-item" data-action="change-colors">' +
        '<span class="ctx-icon">\u{1F3A8}</span>Сменить цвета' +
      '</div>' +
      '<div class="ctx-item" data-action="reset-view">' +
        '<span class="ctx-icon">\u21BA</span>Сбросить масштаб' +
      '</div>' +
      '<div class="ctx-sep"></div>' +
      '<div class="ctx-item" data-action="open-settings">' +
        '<span class="ctx-icon">\u2699</span>Настройки графика' +
      '</div>';

    menu.addEventListener("click", handleItemClick);
    document.body.appendChild(menu);
  }

  /* ---- Позиционирование меню ---- */
  function showMenu(x, y) {
    const menu = document.getElementById("graph-ctx-menu");
    if (!menu) return;

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

  /* ---- Клик по скрытой кнопке Dash ---- */
  function clickButton(id) {
    const btn = document.getElementById(id);
    if (btn) btn.click();
  }

  /* ---- Действия ---- */
  function executeAction(action) {
    switch (action) {
      case "refresh":
        clickButton("update-graf");
        break;

      case "download-html":
        clickButton("download-button");
        break;

      case "download-excel":
        clickButton("download-excel-button");
        break;

      case "copy-png":
        clickButton("copy-png-button");
        break;

      case "change-colors":
        clickButton("shuffle-button");
        break;

      case "reset-view":
        resetGraphView();
        break;

      case "open-settings":
        clickButton("context-menu-btn");
        break;
    }
  }

  /* ---- Сброс масштаба графика ---- */
  function resetGraphView() {
    const host = document.getElementById("graph");
    if (!host) return;
    const gd = host.querySelector(".js-plotly-plot");
    if (!gd || !window.Plotly) return;

    try {
      if (gd.data && gd.data[0] && gd.data[0].type === "scatter3d") {
        Plotly.relayout(gd, {
          "scene.camera.eye": {},
          "scene.camera.center": {},
          "scene.camera.up": {},
        });
      } else {
        const update = {};
        const layout = gd.layout || {};
        for (const key of Object.keys(layout)) {
          if (
            (key.startsWith("xaxis") || key.startsWith("yaxis")) &&
            layout[key] &&
            layout[key].range
          ) {
            update[key + ".range"] = null;
          }
        }
        if (Object.keys(update).length > 0) {
          Plotly.relayout(gd, update);
        }
      }
    } catch (err) {
      console.warn("[ctx-menu] reset error:", err);
    }
  }

  /* ---- Привязка contextmenu к графику ---- */
  function bindGraphContextMenu() {
    const graphHost = document.getElementById("graph");
    if (!graphHost || graphHost.dataset.ctxBound === "1") return;

    graphHost.addEventListener(
      "contextmenu",
      function (e) {
        e.preventDefault();
        e.stopPropagation();
        showMenu(e.clientX, e.clientY);
      },
      true
    );

    graphHost.dataset.ctxBound = "1";
  }

  /* ---- Глобальные обработчики ---- */
  function setupGlobalListeners() {
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
    bindGraphContextMenu();
    setupGlobalListeners();

    new MutationObserver(function () {
      const graphHost = document.getElementById("graph");
      if (graphHost && graphHost.dataset.ctxBound !== "1") {
        bindGraphContextMenu();
      }
    }).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
