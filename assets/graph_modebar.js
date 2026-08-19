/** Добавляет в modebar Plotly кнопки действий рабочей области графика.
 *
 * Через config dcc.Graph передать JS-функцию нельзя, поэтому кнопки
 * вставляются скриптом после рендера графика. Иконки — те же символы,
 * что в тулбаре рабочей области («↻», «⧉»). По клику кнопка программно
 * нажимает соответствующую кнопку рабочей области (data-action-*) —
 * срабатывают те же callback'и, что и для тулбара/контекстного меню.
 */
(function () {
  "use strict";

  var MODEBAR_BUTTONS = [
    { attr: "data-action-refresh", glyph: "↻", title: "Обновить график" },
    { attr: "data-action-copy-png", glyph: "⧉", title: "Копировать PNG в буфер" },
    { attr: "data-action-clear-graph", glyph: "⌫", title: "Очистить график" },
  ];

  function makeButton(spec, targetId) {
    // Родные кнопки Plotly — <button>; тег <a> попал бы под правило
    // ".js-plotly-plot .modebar-group a { display:grid }" и стал бы блочным.
    var button = document.createElement("button");
    button.type = "button";
    button.className = "modebar-btn modebar-btn-custom";
    button.setAttribute("data-title", spec.title);
    button.title = spec.title;
    var icon = document.createElement("span");
    icon.className = "icon modebar-custom-icon";
    icon.textContent = spec.glyph;
    button.appendChild(icon);
    button.addEventListener("click", function (event) {
      event.preventDefault();
      var target = document.getElementById(targetId);
      if (target) {
        target.click();
      }
    });
    return button;
  }

  function decoratePlot(plot) {
    var workspace = plot.closest(".graph-workspace");
    if (!workspace) {
      return;
    }
    var modebar = plot.querySelector(".modebar");
    if (!modebar || modebar.querySelector(".modebar-btn-custom")) {
      return;
    }
    var group = document.createElement("div");
    group.className = "modebar-group modebar-group-custom";
    MODEBAR_BUTTONS.forEach(function (spec) {
      var targetId = workspace.getAttribute(spec.attr);
      if (targetId) {
        group.appendChild(makeButton(spec, targetId));
      }
    });
    if (group.children.length) {
      modebar.insertBefore(group, modebar.firstChild);
    }
  }

  function hookPlot(plot) {
    // При Plotly.react/перерисовке modebar пересоздаётся — возвращаем кнопки.
    if (plot.__modebarCustomHooked || typeof plot.on !== "function") {
      return;
    }
    plot.__modebarCustomHooked = true;
    plot.on("plotly_update", scheduleDecorate);
    plot.on("plotly_redraw", scheduleDecorate);
    plot.on("plotly_relayout", scheduleDecorate);
  }

  function decorateAll(scope) {
    var plots = (scope || document).querySelectorAll(".js-plotly-plot");
    plots.forEach(decoratePlot);
    plots.forEach(hookPlot);
  }

  var scheduled = false;
  function scheduleDecorate() {
    if (scheduled) {
      return;
    }
    scheduled = true;
    requestAnimationFrame(function () {
      scheduled = false;
      decorateAll();
    });
  }

  function looksRelevant(node) {
    return (
      node.nodeType === 1 &&
      (node.matches(".js-plotly-plot, .modebar, .modebar-group") ||
        node.querySelector(".js-plotly-plot, .modebar, .modebar-group"))
    );
  }

  var observer = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i += 1) {
      var added = mutations[i].addedNodes;
      for (var j = 0; j < added.length; j += 1) {
        if (looksRelevant(added[j])) {
          scheduleDecorate();
          return;
        }
      }
    }
  });

  function start() {
    decorateAll();
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
