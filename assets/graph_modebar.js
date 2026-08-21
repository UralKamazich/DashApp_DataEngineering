/** Instance-scoped controls embedded in Plotly's native modebar. */
(function () {
  "use strict";

  var ACTIONS = [
    { attr: "data-action-refresh", glyph: "↻", title: "Обновить график", className: "modebar-refresh-custom" },
    { attr: "data-action-copy-png", glyph: "⧉", title: "Копировать PNG в буфер", className: "modebar-copy-custom" },
    { attr: "data-action-clear-graph", glyph: "⌫", title: "Очистить график" },
  ];

  function makeIconButton(glyph, title, className, onClick) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "modebar-btn modebar-btn-custom " + (className || "");
    button.setAttribute("data-title", title);
    button.setAttribute("aria-label", title);
    button.title = title;

    var icon = document.createElement("span");
    icon.className = "icon modebar-custom-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = glyph;
    button.appendChild(icon);

    if (onClick) {
      button.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        onClick(button);
      });
    }
    return button;
  }

  function clickDashAction(targetId) {
    var target = targetId && document.getElementById(targetId);
    if (target) target.click();
  }

  function chartOptions(workspace) {
    try {
      var options = JSON.parse(
        workspace.getAttribute("data-chart-type-options") || "[]"
      );
      return Array.isArray(options) ? options : [];
    } catch (_error) {
      return [];
    }
  }

  function chartValue(workspace, options) {
    var controlId = workspace.getAttribute("data-chart-type-id");
    var control = controlId && document.getElementById(controlId);
    var raw = control && typeof control.value === "string" ? control.value : "";
    var direct = options.find(function (option) {
      return String(option.value) === raw;
    });
    if (direct) return String(direct.value);
    var byLabel = options.find(function (option) {
      return String(option.label) === raw;
    });
    if (byLabel) return String(byLabel.value);
    return workspace.getAttribute("data-chart-type-value") || "";
  }

  function makeChartSelect(workspace) {
    var options = chartOptions(workspace);
    if (!options.length) return null;

    var select = document.createElement("select");
    select.className = "modebar-chart-select";
    select.setAttribute("aria-label", "Тип графика");
    select.title = "Тип графика";
    options.forEach(function (option) {
      var item = document.createElement("option");
      item.value = String(option.value);
      item.textContent = String(option.label);
      select.appendChild(item);
    });
    select.value = chartValue(workspace, options);

    ["pointerdown", "mousedown", "click"].forEach(function (eventName) {
      select.addEventListener(eventName, function (event) {
        event.stopPropagation();
      });
    });
    select.addEventListener("change", function () {
      var controlId = workspace.getAttribute("data-chart-type-id");
      workspace.setAttribute("data-chart-type-value", select.value);
      if (controlId && window.dash_clientside?.set_props) {
        window.dash_clientside.set_props(controlId, { value: select.value });
      }
    });
    return select;
  }

  function addWorkspaceGroup(plot, modebar, workspace) {
    var existing = modebar.querySelector(".modebar-group-workspace");
    if (existing) {
      var select = existing.querySelector(".modebar-chart-select");
      if (select) select.value = chartValue(workspace, chartOptions(workspace));
      return;
    }

    var group = document.createElement("div");
    group.className = "modebar-group modebar-group-workspace";
    var chartSelect = makeChartSelect(workspace);
    if (chartSelect) group.appendChild(chartSelect);

    if (workspace.getAttribute("data-settings-popup-id")) {
      group.appendChild(makeIconButton("⚙", "Настройки типа графика", "modebar-settings-custom", function (button) {
        if (!window.graphSettingsPopover) return;
        var rect = button.getBoundingClientRect();
        window.graphSettingsPopover.open(
          workspace,
          "specific",
          rect.right,
          rect.bottom
        );
      }));
    }

    var helpId = workspace.getAttribute("data-action-help");
    if (helpId) {
      group.appendChild(makeIconButton("?", "Справка по типу графика", "modebar-help-custom", function () {
        clickDashAction(helpId);
      }));
    }

    ACTIONS.forEach(function (spec) {
      var targetId = workspace.getAttribute(spec.attr);
      if (!targetId) return;
      group.appendChild(makeIconButton(spec.glyph, spec.title, spec.className || "", function () {
        clickDashAction(targetId);
      }));
    });

    if (group.children.length) modebar.insertBefore(group, modebar.firstChild);
  }

  function addFullscreenButton(plot, modebar) {
    var host = plot.closest(".graph-fullscreen-host");
    if (!host || modebar.querySelector(".modebar-fullscreen-custom")) return;

    // graph_fullscreen.js owns the click through event delegation.
    var button = makeIconButton("⛶", "Развернуть график на весь экран", "graph-fullscreen-toggle modebar-fullscreen-custom");
    button.setAttribute("aria-pressed", "false");

    var reset =
      modebar.querySelector('.modebar-btn[data-attr="zoom"][data-val="reset"]') ||
      modebar.querySelector('.modebar-btn[data-attr="resetDefault"]') ||
      modebar.querySelector('[data-title="Reset axes"]') ||
      modebar.querySelector('[data-title="Reset camera to default"]');
    if (reset && reset.parentElement) {
      reset.insertAdjacentElement("afterend", button);
      return;
    }

    var group = document.createElement("div");
    group.className = "modebar-group modebar-group-fullscreen";
    group.appendChild(button);
    modebar.appendChild(group);
  }

  function decoratePlot(plot) {
    var modebar = plot.querySelector(".modebar");
    if (!modebar) return;
    var workspace = plot.closest(".graph-workspace");
    if (workspace) addWorkspaceGroup(plot, modebar, workspace);
    addFullscreenButton(plot, modebar);
  }

  function hookPlot(plot) {
    if (plot.__modebarCustomHooked || typeof plot.on !== "function") return;
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
    if (scheduled) return;
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
