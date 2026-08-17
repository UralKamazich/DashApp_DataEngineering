/**
 * Контекстное меню для графика (правый клик).
 * Перехватывает contextmenu до Plotly и показывает кастомное меню.
 */
(function () {
  "use strict";

  let _menuVisible = false;
  let _lastX = 0;
  let _lastY = 0;

  /* ---- Создание меню (один раз) ---- */
  function createMenu() {
    if (document.getElementById("graph-ctx-menu")) return;

    const menu = document.createElement("div");
    menu.id = "graph-ctx-menu";
    menu.innerHTML =
      '<div class="ctx-item" data-action="open-settings">' +
        '<span class="ctx-icon">\u2699</span>Настройки графика' +
      '</div>' +
      '<div class="ctx-item" data-action="reset-view">' +
        '<span class="ctx-icon">\u21BA</span>Сбросить масштаб' +
      '</div>' +
      '<div class="ctx-sep"></div>' +
      '<div class="ctx-item" data-action="download-png">' +
        '<span class="ctx-icon">\uD83D\uDCF7</span>Скачать как PNG' +
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

  /* ---- Действия ---- */
  function executeAction(action) {
    switch (action) {
      case "open-settings":
        // Кликаем скрытую кнопку — она уже открывает Drawer через callback
        const btn = document.getElementById("context-menu-btn");
        if (btn) btn.click();
        break;

      case "reset-view":
        resetGraphView();
        break;

      case "download-png":
        downloadGraphPng();
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
      // Для 3D-графиков
      if (gd.data && gd.data[0] && gd.data[0].type === "scatter3d") {
        Plotly.relayout(gd, {
          "scene.camera.eye": {},
          "scene.camera.center": {},
          "scene.camera.up": {},
        });
      } else {
        // Для 2D — убираем user-set диапазоны осей
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

  /* ---- Скачать PNG ---- */
  function downloadGraphPng() {
    const host = document.getElementById("graph");
    if (!host) return;
    const gd = host.querySelector(".js-plotly-plot");
    if (!gd || !window.Plotly) return;

    Plotly.toImage(gd, { format: "png", scale: 2 })
      .then(function (dataUrl) {
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = "graph.png";
        document.body.appendChild(a);
        a.click();
        a.remove();
      })
      .catch(function (err) {
        console.error("[ctx-menu] PNG download error:", err);
      });
  }

  /* ---- Привязка contextmenu к графику ---- */
  function bindGraphContextMenu() {
    const graphHost = document.getElementById("graph");
    if (!graphHost || graphHost.dataset.ctxBound === "1") return;

    // Capture-фаза: перехватываем ДО Plotly
    graphHost.addEventListener(
      "contextmenu",
      function (e) {
        e.preventDefault();
        e.stopPropagation();
        _lastX = e.clientX;
        _lastY = e.clientY;
        showMenu(_lastX, _lastY);
      },
      true
    );

    graphHost.dataset.ctxBound = "1";
  }

  /* ---- Глобальные обработчики ---- */
  function setupGlobalListeners() {
    // Клик в любое место — закрыть меню
    document.addEventListener("mousedown", function (e) {
      if (_menuVisible && !e.target.closest("#graph-ctx-menu")) {
        hideMenu();
      }
    });

    // Escape — закрыть меню
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && _menuVisible) {
        hideMenu();
      }
    });

    // Скролл — закрыть меню
    document.addEventListener("scroll", function () {
      if (_menuVisible) hideMenu();
    }, true);
  }

  /* ---- Инициализация ---- */
  function init() {
    createMenu();
    bindGraphContextMenu();
    setupGlobalListeners();

    // MutationObserver: если graph перерендерится, привязка сохраняется
    // благодаря dataset.ctxBound. Но если graph появится заново —
    // пере-привязываем.
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
