/** Drag-and-drop and lightweight interactions for MultiYAxisWorkspace. */
(function () {
  "use strict";

  const WORKSPACE = ".multi-axis-workspace";
  const ZONE = ".multi-axis-drop-zone";
  const COLUMN_MIME = "application/x-dashapp-column";
  const COLORS = [
    "#228be6", "#fa5252", "#40c057", "#fd7e14", "#7950f2",
    "#15aabf", "#e64980", "#82c91e", "#fab005", "#4c6ef5"
  ];
  let draggedColumn = null;

  function parseJson(value, fallback) {
    try {
      const parsed = JSON.parse(value || "");
      return parsed == null ? fallback : parsed;
    } catch (_error) {
      return fallback;
    }
  }

  function copyState(value) {
    const source = value && typeof value === "object" ? value : {};
    try {
      return structuredClone(source);
    } catch (_error) {
      return JSON.parse(JSON.stringify(source));
    }
  }

  function emptyState(datasetId, scope) {
    return {
      dataset_id: datasetId || null,
      scope: scope || "filtered",
      data_ref: null,
      shared_x: null,
      series: [],
      axes: []
    };
  }

  function stateFor(workspace) {
    return copyState(parseJson(
      workspace.getAttribute("data-multi-axis-state"),
      emptyState(
        workspace.getAttribute("data-selected-dataset"),
        workspace.getAttribute("data-selected-scope")
      )
    ));
  }

  function writeState(workspace, state) {
    const storeId = workspace.getAttribute("data-multi-axis-state-id");
    if (!storeId || !window.dash_clientside?.set_props) return;
    workspace.setAttribute("data-multi-axis-state", JSON.stringify(state));
    window.dash_clientside.set_props(storeId, {data: state});
  }

  function setDataset(workspace, datasetId) {
    if (!datasetId || !window.dash_clientside?.set_props) return;
    workspace.setAttribute("data-selected-dataset", datasetId);
    const controlId = workspace.getAttribute("data-multi-axis-dataset-id");
    if (controlId) window.dash_clientside.set_props(controlId, {value: datasetId});
  }

  function columnFromBadge(badge) {
    if (!badge) return null;
    const column = badge.getAttribute("data-column-name");
    if (!column) return null;
    return {
      column: column,
      type: badge.getAttribute("data-column-type") || "",
      datasetId: badge.getAttribute("data-dataset-id") || ""
    };
  }

  function columnFromTransfer(event) {
    const custom = parseJson(event.dataTransfer?.getData(COLUMN_MIME), null);
    if (custom?.column) return custom;
    const plain = event.dataTransfer?.getData("text/plain");
    if (plain) {
      return Object.assign({}, draggedColumn || {}, {column: plain});
    }
    return draggedColumn;
  }

  function accepts(zone, column) {
    if (!zone || !column?.column) return false;
    const kind = zone.getAttribute("data-multi-axis-drop");
    return kind === "x" || (kind === "y" && column.type === "numeric");
  }

  function makeId(prefix) {
    const random = Math.random().toString(36).slice(2, 7);
    return prefix + "-" + Date.now().toString(36) + "-" + random;
  }

  function addSeries(state, column, side) {
    const seriesId = makeId("series");
    const axisId = "axis-" + seriesId;
    const seriesCount = Array.isArray(state.series) ? state.series.length : 0;
    const color = COLORS[seriesCount % COLORS.length];
    const usedRefs = new Set((Array.isArray(state.axes) ? state.axes : []).map(function (axis) {
      return String(axis.plotly_ref || "");
    }));
    let refNumber = 2;
    while (usedRefs.has("y" + refNumber)) refNumber += 1;

    state.series = Array.isArray(state.series) ? state.series : [];
    state.axes = Array.isArray(state.axes) ? state.axes : [];
    state.series.push({
      id: seriesId,
      y: column,
      x: null,
      x_mode: "shared",
      type: "line",
      name: column,
      color: color,
      side: side,
      axis_id: axisId,
      visible: true
    });
    state.axes.push({
      id: axisId,
      plotly_ref: "y" + refNumber,
      title: column,
      side: side,
      type: "linear",
      autorange: true,
      range: null,
      visible: true
    });
    return state;
  }

  function applyDrop(zone, column) {
    const workspace = zone.closest(WORKSPACE);
    if (!workspace || !accepts(zone, column)) return false;

    const selectedDataset = workspace.getAttribute("data-selected-dataset") || "";
    const stateDataset = String(stateFor(workspace).dataset_id || selectedDataset || "");
    const sourceDataset = String(column.datasetId || "");
    const scope = workspace.getAttribute("data-selected-scope") || "filtered";
    let state = stateFor(workspace);

    // A workspace cannot mix channels from different dataframes.  A drop from
    // another dataset explicitly switches this instance and clears old pairs.
    if (sourceDataset && stateDataset && sourceDataset !== stateDataset) {
      state = emptyState(sourceDataset, scope);
    }
    if (sourceDataset) state.dataset_id = sourceDataset;
    state.scope = state.scope || scope;

    if (zone.getAttribute("data-multi-axis-drop") === "x") {
      state.shared_x = column.column;
    } else {
      state = addSeries(
        state,
        column.column,
        zone.getAttribute("data-axis-side") === "right" ? "right" : "left"
      );
    }

    writeState(workspace, state);
    if (sourceDataset) setDataset(workspace, sourceDataset);
    return true;
  }

  function clearHover(workspace) {
    workspace?.querySelectorAll(ZONE).forEach(function (zone) {
      zone.classList.remove("zone-hover", "zone-rejected");
    });
  }

  function setDragging(active) {
    document.querySelectorAll(WORKSPACE).forEach(function (workspace) {
      workspace.classList.toggle("multi-axis-dnd-active", active);
      if (!active) clearHover(workspace);
    });
  }

  function openSeriesCard(chip) {
    const workspace = chip.closest(WORKSPACE);
    if (!workspace) return;
    const seriesId = chip.getAttribute("data-series-id");
    const popupId = workspace.getAttribute("data-settings-popup-id");
    const popup = popupId && document.getElementById(popupId);
    if (!popup) return;

    const rect = chip.getBoundingClientRect();
    if (window.graphSettingsPopover) {
      window.graphSettingsPopover.open(workspace, "specific", rect.right, rect.bottom);
    }
    requestAnimationFrame(function () {
      const card = Array.from(popup.querySelectorAll(".multi-axis-series-card")).find(function (item) {
        return item.getAttribute("data-series-id") === seriesId;
      });
      if (!card) return;
      card.scrollIntoView({block: "nearest", behavior: "smooth"});
      card.classList.remove("is-highlighted");
      void card.offsetWidth;
      card.classList.add("is-highlighted");
      window.setTimeout(function () { card.classList.remove("is-highlighted"); }, 900);
    });
  }

  document.addEventListener("dragstart", function (event) {
    const badge = event.target.closest("[data-column-name]");
    draggedColumn = columnFromBadge(badge);
    if (!draggedColumn) return;
    try {
      event.dataTransfer.setData(COLUMN_MIME, JSON.stringify(draggedColumn));
    } catch (_error) {
      // text/plain is already supplied by the dataset badge's shared handler.
    }
    setDragging(true);
  });

  document.addEventListener("dragend", function () {
    draggedColumn = null;
    setDragging(false);
  });

  document.addEventListener("dragover", function (event) {
    const zone = event.target.closest(ZONE);
    if (!zone) return;
    event.preventDefault();
    const column = draggedColumn || columnFromTransfer(event);
    const allowed = accepts(zone, column);
    clearHover(zone.closest(WORKSPACE));
    zone.classList.toggle("zone-hover", allowed);
    zone.classList.toggle("zone-rejected", !allowed);
    if (event.dataTransfer) event.dataTransfer.dropEffect = allowed ? "copy" : "none";
  });

  document.addEventListener("dragleave", function (event) {
    const zone = event.target.closest(ZONE);
    if (!zone) return;
    if (!event.relatedTarget || !zone.contains(event.relatedTarget)) {
      zone.classList.remove("zone-hover", "zone-rejected");
    }
  });

  document.addEventListener("drop", function (event) {
    const zone = event.target.closest(ZONE);
    if (!zone) return;
    event.preventDefault();
    event.stopPropagation();
    const column = columnFromTransfer(event);
    applyDrop(zone, column);
    draggedColumn = null;
    setDragging(false);
  });

  document.addEventListener("click", function (event) {
    const clearButton = event.target.closest(".multi-axis-drop-clear");
    if (clearButton) {
      const workspace = clearButton.closest(WORKSPACE);
      const state = stateFor(workspace);
      state.shared_x = null;
      writeState(workspace, state);
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    const chip = event.target.closest(".multi-axis-series-chip");
    if (chip) {
      openSeriesCard(chip);
      event.preventDefault();
      event.stopPropagation();
    }
  });

  window.multiAxisWorkspace = Object.freeze({
    emptyState: emptyState,
    applyDrop: applyDrop
  });
})();
