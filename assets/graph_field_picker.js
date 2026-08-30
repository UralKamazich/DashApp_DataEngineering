/** Searchable field picker opened from a GraphWorkspace drop zone. */
(function () {
  "use strict";

  let activeZone = null;
  let draggedBadge = null;
  let datasetColumns = [];
  let sortColumns = false;
  const dropTargetSelector = ".graph-drop-zone, .correlation-channel-drop";
  const categoricalFields = new Set([
    "x", "y", "z", "color", "text", "facet-row", "facet-col", "hover"
  ]);
  const alwaysCategoricalFields = new Set(["hierarchy-levels"]);

  function readZoneValue(zone) {
    try {
      return JSON.parse(zone.getAttribute("data-current-value") || "null");
    } catch (_error) {
      return null;
    }
  }

  function writeZoneValue(zone, value) {
    const targetId = zone.getAttribute("data-drop-target");
    if (!targetId) return;
    zone.setAttribute("data-current-value", JSON.stringify(value));
    window.dash_clientside.set_props(targetId, { value });
  }

  function readFieldModes(zone) {
    const workspace = zone?.closest(".graph-workspace");
    try {
      return JSON.parse(workspace?.getAttribute("data-field-modes") || "{}");
    } catch (_error) {
      return {};
    }
  }

  function writeFieldMode(zone, asCategorical) {
    const workspace = zone?.closest(".graph-workspace");
    const fieldKey = zone?.getAttribute("data-field-key");
    const storeId = workspace?.getAttribute("data-field-mode-store-id");
    if (!workspace || !fieldKey || !storeId) return;

    const modes = readFieldModes(zone);
    if (asCategorical) modes[fieldKey] = true;
    else delete modes[fieldKey];
    workspace.setAttribute("data-field-modes", JSON.stringify(modes));
    zone.classList.toggle("as-categorical", Boolean(asCategorical));
    if (window.dash_clientside?.set_props) {
      window.dash_clientside.set_props(storeId, { data: modes });
    }
  }

  function getDatasetColumns(zone) {
    const catalog = document.getElementById("dataset-column-catalog");
    if (catalog && catalog.textContent) {
      try {
        const columns = JSON.parse(catalog.textContent);
        if (Array.isArray(columns)) {
          return columns.map(String);
        }
      } catch (_error) {
        // Compatibility fallback: collect the currently rendered badges.
      }
    }
    const workspace = zone && zone.closest(".graph-workspace");
    const containerId = workspace && workspace.getAttribute("data-columns-container-id");
    const container = (containerId && document.getElementById(containerId)) || document;
    const seen = new Set();
    return Array.from(
      container.querySelectorAll("[data-column-name]")
    ).reduce(function (columns, badge) {
      const name = badge.getAttribute("data-column-name");
      if (name && !seen.has(name)) {
        seen.add(name);
        columns.push(name);
      }
      return columns;
    }, []);
  }

  function createPicker() {
    if (document.getElementById("graph-field-picker")) return;

    const picker = document.createElement("section");
    picker.id = "graph-field-picker";
    picker.setAttribute("role", "dialog");
    picker.setAttribute("aria-modal", "false");
    picker.setAttribute("aria-labelledby", "graph-field-picker-title");
    picker.innerHTML =
      '<header class="field-picker-header">' +
        '<div>' +
          '<div class="field-picker-kicker">Поле графика</div>' +
          '<div id="graph-field-picker-title" class="field-picker-title"></div>' +
        '</div>' +
        '<button type="button" class="field-picker-close" aria-label="Закрыть" title="Закрыть">×</button>' +
      '</header>' +
      '<div class="field-picker-tools">' +
        '<label class="field-picker-search-wrap">' +
          '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>' +
          '<input class="field-picker-search" type="search" placeholder="Поиск столбца" autocomplete="off" spellcheck="false">' +
        '</label>' +
        '<div class="field-picker-controls-row">' +
          '<div class="field-picker-order">' +
            '<span class="field-picker-order-label is-active" data-order-label="dataset">Dataset</span>' +
            '<button type="button" class="field-picker-switch field-picker-sort-switch" role="switch" aria-checked="false" aria-label="Сортировка столбцов">' +
              '<span class="field-picker-switch-knob"></span>' +
            '</button>' +
            '<span class="field-picker-order-label" data-order-label="sort">Sort</span>' +
          '</div>' +
          '<div class="field-picker-category-mode">' +
            '<span class="field-picker-category-label is-active" data-category-label="dataset">Dataset</span>' +
            '<button type="button" class="field-picker-switch field-picker-category-switch" role="switch" aria-checked="false" aria-label="Использовать как категорию">' +
              '<span class="field-picker-switch-knob"></span>' +
            '</button>' +
            '<span class="field-picker-category-label" data-category-label="categorical">As categorical</span>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<div class="field-picker-list" role="listbox"></div>' +
      '<footer class="field-picker-footer">' +
        '<span>ПКМ по другой зоне переключит поле</span>' +
        '<button type="button" class="field-picker-clear">Очистить</button>' +
      '</footer>';

    picker.querySelector(".field-picker-close").addEventListener("click", closePicker);
    picker.querySelector(".field-picker-clear").addEventListener("click", function () {
      if (!activeZone) return;
      clearZone(activeZone);
      closePicker();
    });
    picker.querySelector(".field-picker-search").addEventListener("input", renderColumns);
    picker.querySelector(".field-picker-sort-switch").addEventListener("click", function () {
      sortColumns = !sortColumns;
      updateSortControl();
      renderColumns();
    });
    picker.querySelector(".field-picker-category-switch").addEventListener("click", function () {
      if (!activeZone || this.disabled) return;
      const fieldKey = activeZone.getAttribute("data-field-key");
      const modes = readFieldModes(activeZone);
      writeFieldMode(activeZone, !modes[fieldKey]);
      updateCategoryControl();
    });
    picker.querySelector(".field-picker-list").addEventListener("click", function (event) {
      const option = event.target.closest("[data-column-option]");
      if (!option || !activeZone) return;
      selectColumn(option.getAttribute("data-column-option"));
    });

    document.body.appendChild(picker);
  }

  function updateSortControl() {
    const picker = document.getElementById("graph-field-picker");
    const toggle = picker.querySelector(".field-picker-sort-switch");
    toggle.setAttribute("aria-checked", String(sortColumns));
    picker.querySelector('[data-order-label="dataset"]').classList.toggle("is-active", !sortColumns);
    picker.querySelector('[data-order-label="sort"]').classList.toggle("is-active", sortColumns);
  }

  function updateCategoryControl() {
    const picker = document.getElementById("graph-field-picker");
    const toggle = picker.querySelector(".field-picker-category-switch");
    const fieldKey = activeZone?.getAttribute("data-field-key");
    const supported = categoricalFields.has(fieldKey);
    const alwaysCategorical = alwaysCategoricalFields.has(fieldKey);
    const checked = alwaysCategorical || (supported && Boolean(readFieldModes(activeZone)[fieldKey]));
    toggle.disabled = alwaysCategorical || !supported;
    toggle.setAttribute("aria-checked", String(checked));
    toggle.title = alwaysCategorical
      ? "Уровни иерархии всегда отображаются как категории"
      : supported
      ? "Временно показать значения выбранного поля как категории"
      : "Категориальный режим неприменим к размеру маркера";
    picker.querySelector('[data-category-label="dataset"]').classList.toggle("is-active", !checked);
    picker.querySelector('[data-category-label="categorical"]').classList.toggle("is-active", checked);
    picker.querySelector(".field-picker-category-mode").classList.toggle(
      "is-disabled", alwaysCategorical || !supported
    );
  }

  function renderColumns() {
    const picker = document.getElementById("graph-field-picker");
    if (!picker || !activeZone) return;

    const search = picker.querySelector(".field-picker-search").value.trim().toLocaleLowerCase("ru");
    const current = readZoneValue(activeZone);
    const selected = new Set(Array.isArray(current) ? current : current ? [current] : []);
    let columns = datasetColumns.filter(function (column) {
      return !search || column.toLocaleLowerCase("ru").includes(search);
    });

    if (sortColumns) {
      columns = columns.slice().sort(function (left, right) {
        return left.localeCompare(right, "ru", { numeric: true, sensitivity: "base" });
      });
    }

    const list = picker.querySelector(".field-picker-list");
    list.replaceChildren();

    if (!columns.length) {
      const empty = document.createElement("div");
      empty.className = "field-picker-empty";
      empty.textContent = datasetColumns.length
        ? "Столбцы не найдены"
        : "В датасете пока нет столбцов";
      list.appendChild(empty);
      return;
    }

    columns.forEach(function (column) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "field-picker-option";
      option.setAttribute("role", "option");
      option.setAttribute("data-column-option", column);
      option.setAttribute("aria-selected", String(selected.has(column)));

      const name = document.createElement("span");
      name.className = "field-picker-option-name";
      name.textContent = column;
      option.appendChild(name);

      const check = document.createElement("span");
      check.className = "field-picker-option-check";
      check.textContent = "✓";
      check.setAttribute("aria-hidden", "true");
      option.appendChild(check);
      list.appendChild(option);
    });
  }

  function selectColumn(column) {
    if (!activeZone || !column) return;
    const mode = activeZone.getAttribute("data-drop-mode") || "replace";
    const current = readZoneValue(activeZone);

    if (mode === "append") {
      const values = Array.isArray(current) ? current.slice() : [];
      const index = values.indexOf(column);
      if (index === -1) values.push(column);
      else values.splice(index, 1);
      writeZoneValue(activeZone, values);
      renderColumns();
      return;
    }

    writeZoneValue(activeZone, column);
    closePicker();
  }

  function positionPicker(x, y) {
    const picker = document.getElementById("graph-field-picker");
    const gap = 12;
    const edge = 8;
    const width = picker.offsetWidth;
    const height = picker.offsetHeight;
    let left = x + gap;
    let top = y - 12;

    if (left + width > window.innerWidth - edge) {
      left = Math.max(edge, x - width - gap);
    }
    top = Math.min(Math.max(edge, top), Math.max(edge, window.innerHeight - height - edge));
    picker.style.left = left + "px";
    picker.style.top = top + "px";
  }

  function pickerPortal(zone) {
    const fullscreenHost = document.fullscreenElement;
    if (fullscreenHost?.contains(zone)) return fullscreenHost;
    return document.body;
  }

  function openPicker(zone, x, y) {
    createPicker();
    activeZone = zone;
    datasetColumns = getDatasetColumns(zone);

    const picker = document.getElementById("graph-field-picker");
    const portal = pickerPortal(zone);
    if (picker.parentElement !== portal) portal.appendChild(picker);
    picker.querySelector(".field-picker-title").textContent =
      zone.getAttribute("data-active-label") ||
      zone.getAttribute("data-default-label") || "Столбец";
    picker.querySelector(".field-picker-search").value = "";
    picker.classList.add("is-open");
    zone.closest(".graph-workspace")?.classList.add("field-picker-open");
    updateSortControl();
    updateCategoryControl();
    renderColumns();
    picker.querySelector(".field-picker-clear").disabled = !readZoneValue(zone) ||
      (Array.isArray(readZoneValue(zone)) && !readZoneValue(zone).length);
    positionPicker(x, y);
    picker.querySelector(".field-picker-search").focus({ preventScroll: true });
  }

  function closePicker() {
    const picker = document.getElementById("graph-field-picker");
    if (picker) picker.classList.remove("is-open");
    activeZone?.closest(".graph-workspace")?.classList.remove("field-picker-open");
    activeZone = null;
  }

  function clearZone(zone) {
    const emptyValue = zone.getAttribute("data-drop-mode") === "append" ? [] : null;
    writeZoneValue(zone, emptyValue);
    writeFieldMode(zone, false);
    if (zone === activeZone) renderColumns();
  }

  function setDragging(active) {
    document.querySelectorAll(".graph-workspace").forEach(function (workspace) {
      if (active) syncAxisDropZones(workspace);
      workspace.classList.toggle("dnd-active", active);
      if (!active) {
        workspace.querySelectorAll(".zone-hover").forEach(function (zone) {
          zone.classList.remove("zone-hover");
        });
        workspace.classList.remove("axis-x-drop-hover", "axis-y-drop-hover");
      }
    });
    document.querySelectorAll(".correlation-channel-drop").forEach(function (target) {
      target.classList.toggle("dnd-active", active);
      if (!active) {
        target.classList.remove("zone-hover", "zone-rejected");
      }
    });
  }

  function acceptsDraggedColumn(zone) {
    const acceptedType = zone.getAttribute("data-accept-type");
    if (!acceptedType) return true;
    return draggedBadge?.getAttribute("data-column-type") === acceptedType;
  }

  function setDroppedField(zone, columnName) {
    if (!columnName) return;
    const mode = zone.getAttribute("data-drop-mode") || "replace";
    const current = readZoneValue(zone);
    if (mode === "append") {
      const values = Array.isArray(current) ? current.slice() : [];
      const index = values.indexOf(columnName);
      if (index === -1) values.push(columnName);
      else values.splice(index, 1);
      writeZoneValue(zone, values);
      return;
    }
    writeZoneValue(zone, current === columnName ? null : columnName);
  }

  function plotGeometry(workspace) {
    const gd = workspace?.querySelector(".js-plotly-plot");
    const size = gd?._fullLayout?._size;
    if (!gd || !size) return null;
    const workspaceRect = workspace.getBoundingClientRect();
    const graphRect = gd.getBoundingClientRect();
    return {gd, size, workspaceRect, graphRect};
  }

  function axisZone(workspace, fieldKey) {
    return workspace?.querySelector('.graph-drop-zone[data-field-key="' + fieldKey + '"]');
  }

  function supportedChartTypes(zone) {
    const raw = zone?.getAttribute("data-chart-types");
    if (raw == null) return null;
    try {
      const values = JSON.parse(raw);
      return Array.isArray(values) ? values.map(String) : [];
    } catch (_error) {
      return [];
    }
  }

  function fieldPresentations(workspace) {
    try {
      const value = JSON.parse(
        workspace?.getAttribute("data-field-presentations") || "{}"
      );
      return value && typeof value === "object" ? value : {};
    } catch (_error) {
      return {};
    }
  }

  function applyFieldPresentation(workspace, chartType) {
    const presentations = fieldPresentations(workspace);
    const chartPresentation = presentations[String(chartType || "")] || {};
    const topZones = [];

    workspace.querySelectorAll(".graph-drop-zone").forEach(function (zone) {
      const fieldKey = zone.getAttribute("data-field-key");
      const presentation = chartPresentation[fieldKey] || {};
      const label = presentation.label || zone.getAttribute("data-default-label") || fieldKey;
      const labelElement = zone.querySelector(".graph-drop-zone-name");
      if (labelElement) labelElement.textContent = label;
      zone.setAttribute("data-active-label", label);
      zone.classList.toggle("graph-drop-zone--top-channel", presentation.placement === "top");
      zone.style.removeProperty("--graph-top-left");
      zone.style.order = presentation.order == null ? "" : String(presentation.order);
      if (presentation.placement === "top" && !zone.hidden) topZones.push(zone);
    });

    topZones.forEach(function (zone, index) {
      zone.style.setProperty("--graph-top-left", (12 + index * 103) + "px");
    });
    workspace.classList.toggle("has-top-channels", topZones.length > 0);
    workspace.classList.toggle(
      "has-axisless-channels",
      ["3D_Scatter", "Polar", "Pie", "Sunburst", "Treemap"].includes(String(chartType || ""))
    );
    workspace.style.setProperty(
      "--graph-aux-start",
      (topZones.length ? 12 + topZones.length * 103 : 12) + "px"
    );
  }

  function updateFieldAvailability(workspace, chartType) {
    if (!workspace) return;
    const selectedType = String(
      chartType || workspace.getAttribute("data-chart-type-value") || ""
    );
    workspace.querySelectorAll(".graph-drop-zone[data-chart-types]").forEach(function (zone) {
      const supported = supportedChartTypes(zone) || [];
      const available = Boolean(selectedType) && supported.includes(selectedType);
      zone.hidden = !available;
      zone.classList.toggle("is-chart-incompatible", !available);
      if (!available) zone.classList.remove("zone-hover", "zone-rejected");
    });
    applyFieldPresentation(workspace, selectedType);
    const zZone = axisZone(workspace, "z");
    workspace.classList.toggle("aux-z-visible", Boolean(zZone && !zZone.hidden));
    if (activeZone?.hidden) closePicker();
  }

  function syncAxisDropZones(workspace) {
    const geometry = plotGeometry(workspace);
    if (!geometry) return;
    const {size, workspaceRect, graphRect} = geometry;
    const xZone = axisZone(workspace, "x");
    const yZone = axisZone(workspace, "y");
    const plotLeft = graphRect.left - workspaceRect.left + size.l;
    const plotTop = graphRect.top - workspaceRect.top + size.t;

    if (xZone && !xZone.classList.contains("graph-drop-zone--top-channel")) {
      xZone.style.left = plotLeft + "px";
      xZone.style.top = (plotTop + size.h - 7) + "px";
      xZone.style.width = size.w + "px";
      xZone.style.height = Math.max(34, Math.min(58, graphRect.bottom - (graphRect.top + size.t + size.h) + 10)) + "px";
    }
    if (yZone && !yZone.classList.contains("graph-drop-zone--top-channel")) {
      yZone.style.left = Math.max(0, plotLeft - 72) + "px";
      yZone.style.top = plotTop + "px";
      yZone.style.width = Math.min(80, plotLeft + 10) + "px";
      yZone.style.height = size.h + "px";
    }
  }

  function axisFieldFromTarget(target) {
    if (!(target instanceof Element)) return null;
    if (target.closest(".xaxislayer-above, .xaxislayer-below, .g-xtitle, .xtick, .xlines-above, .xlines-below")) {
      return "x";
    }
    if (target.closest(".yaxislayer-above, .yaxislayer-below, .g-ytitle, .ytick, .ylines-above, .ylines-below")) {
      return "y";
    }
    return null;
  }

  function axisFieldFromPoint(workspace, x, y) {
    const geometry = plotGeometry(workspace);
    if (!geometry) return null;
    const {size, graphRect} = geometry;
    const plotLeft = graphRect.left + size.l;
    const plotTop = graphRect.top + size.t;
    const plotBottom = plotTop + size.h;
    if (x >= plotLeft - 72 && x <= plotLeft + 9 && y >= plotTop && y <= plotBottom) return "y";
    if (x >= plotLeft && x <= plotLeft + size.w && y >= plotBottom - 7 && y <= plotBottom + 55) return "x";
    return null;
  }

  function openAxisPickerForEvent(event) {
    const workspace = event.target?.closest?.(".graph-workspace");
    if (!workspace) return false;
    if (workspace.classList.contains("has-axisless-channels")) return false;
    const fieldKey = axisFieldFromTarget(event.target) ||
      axisFieldFromPoint(workspace, event.clientX, event.clientY);
    const zone = fieldKey && axisZone(workspace, fieldKey);
    if (!zone) return false;
    openPicker(zone, event.clientX, event.clientY);
    return true;
  }

  function bindAxisZoneSync(workspace) {
    const gd = workspace?.querySelector(".js-plotly-plot");
    if (!gd || typeof gd.on !== "function") return false;
    if (gd.dataset.graphAxisDropSyncBound === "true") return true;
    gd.dataset.graphAxisDropSyncBound = "true";
    gd.on("plotly_afterplot", function () {
      requestAnimationFrame(function () { syncAxisDropZones(workspace); });
    });
    requestAnimationFrame(function () { syncAxisDropZones(workspace); });
    return true;
  }

  function discoverAxisZonePlots() {
    document.querySelectorAll(".graph-workspace").forEach(function (workspace) {
      bindAxisZoneSync(workspace);
      updateFieldAvailability(workspace);
    });
  }

  function makeDragPreview(columnName) {
    const preview = document.createElement("div");
    preview.className = "column-drag-preview";
    preview.textContent = columnName;
    document.body.appendChild(preview);
    return preview;
  }

  function installListeners() {
    document.addEventListener("dragstart", function (event) {
      const badge = event.target.closest("[data-column-name]");
      if (!badge) return;
      const columnName = badge.getAttribute("data-column-name");
      if (!columnName) return;

      closePicker();
      draggedBadge = badge;
      event.dataTransfer.setData("text/plain", columnName);
      event.dataTransfer.effectAllowed = "copy";
      const preview = makeDragPreview(columnName);
      event.dataTransfer.setDragImage(preview, 16, 16);
      window.setTimeout(function () { preview.remove(); }, 0);
      badge.classList.add("column-badge--dragging");
      setDragging(true);
    });

    document.addEventListener("dragend", function () {
      if (draggedBadge) draggedBadge.classList.remove("column-badge--dragging");
      draggedBadge = null;
      setDragging(false);
    });

    document.addEventListener("dragover", function (event) {
      const zone = event.target.closest(dropTargetSelector);
      if (!zone) return;
      event.preventDefault();
      const accepted = acceptsDraggedColumn(zone);
      event.dataTransfer.dropEffect = accepted ? "copy" : "none";
      const workspace = zone.closest(".graph-workspace");
      const scope = workspace || zone.parentElement || document;
      scope.querySelectorAll(".zone-hover, .zone-rejected").forEach(function (item) {
        if (item !== zone) item.classList.remove("zone-hover", "zone-rejected");
      });
      zone.classList.toggle("zone-hover", accepted);
      zone.classList.toggle("zone-rejected", !accepted);
      const fieldKey = zone.getAttribute("data-field-key");
      if (workspace && (fieldKey === "x" || fieldKey === "y")) {
        workspace.classList.toggle("axis-" + fieldKey + "-drop-hover", accepted);
      }
    });

    document.addEventListener("dragleave", function (event) {
      const zone = event.target.closest(dropTargetSelector);
      if (!zone) return;
      if (!event.relatedTarget || !zone.contains(event.relatedTarget)) {
        zone.classList.remove("zone-hover", "zone-rejected");
        const workspace = zone.closest(".graph-workspace");
        const fieldKey = zone.getAttribute("data-field-key");
        if (workspace && (fieldKey === "x" || fieldKey === "y")) {
          workspace.classList.remove("axis-" + fieldKey + "-drop-hover");
        }
      }
    });

    document.addEventListener("drop", function (event) {
      const zone = event.target.closest(dropTargetSelector);
      if (!zone) return;
      event.preventDefault();
      event.stopPropagation();
      if (acceptsDraggedColumn(zone)) {
        setDroppedField(zone, event.dataTransfer.getData("text/plain"));
      }
      if (draggedBadge) draggedBadge.classList.remove("column-badge--dragging");
      draggedBadge = null;
      setDragging(false);
    });

    document.addEventListener("contextmenu", function (event) {
      const zone = event.target.closest(".graph-drop-zone");
      if (!zone) {
        if (!openAxisPickerForEvent(event)) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }
      event.preventDefault();
      event.stopImmediatePropagation();
      openPicker(zone, event.clientX, event.clientY);
    }, true);

    document.addEventListener("click", function (event) {
      const clearButton = event.target.closest(".graph-zone-clear");
      if (!clearButton) return;
      const zone = clearButton.closest(".graph-drop-zone");
      if (!zone) return;
      event.preventDefault();
      event.stopPropagation();
      clearZone(zone);
    });

    document.addEventListener("pointerdown", function (event) {
      const picker = document.getElementById("graph-field-picker");
      if (!picker?.classList.contains("is-open")) return;
      if (!picker.contains(event.target)) closePicker();
    }, true);

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") closePicker();
    });

    window.addEventListener("resize", closePicker);
    window.addEventListener("resize", discoverAxisZonePlots);
  }

  function init() {
    createPicker();
    installListeners();
    discoverAxisZonePlots();
    const observer = new MutationObserver(discoverAxisZonePlots);
    observer.observe(document.documentElement, {childList: true, subtree: true});
    window.graphFieldPicker = Object.assign(window.graphFieldPicker || {}, {
      openAxisPickerForEvent: openAxisPickerForEvent,
      syncAxisDropZones: syncAxisDropZones,
      updateFieldAvailability: updateFieldAvailability
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
