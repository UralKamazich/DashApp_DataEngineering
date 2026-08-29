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
      show_legend: true,
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
      type: "scatter",
      name: column,
      color: color,
      smooth: false,
      side: side,
      axis_id: axisId,
      visible: true
    });
    state.axes.push({
      id: axisId,
      plotly_ref: "y" + refNumber,
      side: side,
      type: "linear"
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

  function linearRange(axis) {
    if (!axis || !Array.isArray(axis.range) || axis.range.length < 2) return null;
    return axis.range.slice(0, 2).map(function (value) {
      return typeof axis.r2l === "function" ? axis.r2l(value) : Number(value);
    });
  }

  function dataRange(axis, values) {
    return values.map(function (value) {
      return typeof axis.l2r === "function" ? axis.l2r(value) : value;
    });
  }

  function relayoutAxis(gd, axisName, range) {
    if (!window.Plotly || !gd || !axisName || !range) return;
    const update = {};
    update[axisName + ".autorange"] = false;
    update[axisName + ".range"] = range;
    window.Plotly.relayout(gd, update);
  }

  function axisTickBounds(gd, axis) {
    const tickClass = String(axis?._id || "") + "tick";
    if (!tickClass) return null;
    const ticks = Array.from(gd.querySelectorAll("g." + tickClass));
    if (!ticks.length) return null;
    const rects = ticks.map(function (tick) { return tick.getBoundingClientRect(); })
      .filter(function (rect) { return rect.width > 0 && rect.height > 0; });
    if (!rects.length) return null;
    return rects.reduce(function (bounds, rect) {
      return {
        left: Math.min(bounds.left, rect.left),
        right: Math.max(bounds.right, rect.right),
        top: Math.min(bounds.top, rect.top),
        bottom: Math.max(bounds.bottom, rect.bottom)
      };
    }, {left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity});
  }

  function axisTitleBounds(gd, axis) {
    const title = gd.querySelector(".g-" + String(axis?._id || "") + "title");
    if (!title) return null;
    const rect = title.getBoundingClientRect();
    if (!Number.isFinite(rect.width) || rect.width <= 0) return null;
    return rect;
  }

  function almostEqual(first, second, tolerance) {
    return Math.abs(Number(first) - Number(second)) <= (tolerance || 0.001);
  }

  function syncAxisRailLayout(workspace, gd) {
    const layout = gd?._fullLayout;
    if (!layout || !window.Plotly || workspace.dataset.multiAxisRailRelayout === "true") {
      return false;
    }
    const entries = Object.keys(layout).filter(function (key) {
      return /^yaxis\d*$/.test(key);
    }).map(function (axisName) {
      const axis = layout[axisName];
      if (!axis || axis.visible === false || axis._id === "y") return null;
      const bounds = axisTickBounds(gd, axis);
      if (!bounds) return null;
      return {
        axisName: axisName,
        axis: axis,
        bounds: bounds,
        titleBounds: axisTitleBounds(gd, axis)
      };
    }).filter(Boolean);
    if (!entries.length) return false;

    const activeNames = new Set(entries.map(function (entry) { return entry.axisName; }));
    const cached = workspace._multiAxisRailWidths || {};
    Object.keys(cached).forEach(function (axisName) {
      if (!activeNames.has(axisName)) delete cached[axisName];
    });
    entries.forEach(function (entry) {
      const bounds = entry.bounds;
      const titleBounds = entry.titleBounds;
      const lineX = entry.axis.side === "right" ? bounds.left - 5 : bounds.right + 5;
      const outerEdge = entry.axis.side === "right"
        ? Math.max(bounds.right, titleBounds?.right || bounds.right)
        : Math.min(bounds.left, titleBounds?.left || bounds.left);
      const envelope = entry.axis.side === "right" ? outerEdge - lineX : lineX - outerEdge;
      const measured = Math.max(60, Math.min(128, envelope + 10));
      cached[entry.axisName] = Math.max(Number(cached[entry.axisName]) || 0, measured);
      entry.railWidth = cached[entry.axisName];
    });
    workspace._multiAxisRailWidths = cached;

    const left = entries.filter(function (entry) { return entry.axis.side !== "right"; })
      .sort(function (a, b) { return b.axis.position - a.axis.position; });
    const right = entries.filter(function (entry) { return entry.axis.side === "right"; })
      .sort(function (a, b) { return a.axis.position - b.axis.position; });
    const leftOuterWidth = left.length ? left[left.length - 1].railWidth : 0;
    const rightOuterWidth = right.length ? right[right.length - 1].railWidth : 0;
    const marginLeft = Math.ceil(Math.max(60, leftOuterWidth + 10));
    const marginRight = Math.ceil(Math.max(45, rightOuterWidth + 10));
    const innerWidth = Math.max(280, gd.clientWidth - marginLeft - marginRight);
    const leftReserve = left.slice(0, -1).reduce(function (sum, entry) {
      return sum + entry.railWidth;
    }, 0);
    const rightReserve = right.slice(0, -1).reduce(function (sum, entry) {
      return sum + entry.railWidth;
    }, 0);
    const domainLeft = leftReserve / innerWidth;
    const domainRight = 1 - rightReserve / innerWidth;
    if (domainRight - domainLeft < 0.22) {
      // Keep a minimal useful data surface. This only affects extreme cases
      // where the container is physically too narrow for all measured rails.
      const scale = Math.max(0, (innerWidth * 0.78) / Math.max(1, leftReserve + rightReserve));
      left.forEach(function (entry) { entry.railWidth *= Math.min(1, scale); });
      right.forEach(function (entry) { entry.railWidth *= Math.min(1, scale); });
    }
    const adjustedLeftReserve = left.slice(0, -1).reduce(function (sum, entry) {
      return sum + entry.railWidth;
    }, 0);
    const adjustedRightReserve = right.slice(0, -1).reduce(function (sum, entry) {
      return sum + entry.railWidth;
    }, 0);
    const adjustedDomainLeft = adjustedLeftReserve / innerWidth;
    const adjustedDomainRight = 1 - adjustedRightReserve / innerWidth;
    const updates = {
      "margin.l": marginLeft,
      "margin.r": marginRight,
      "xaxis.domain": [adjustedDomainLeft, adjustedDomainRight]
    };

    let leftOffset = 0;
    left.forEach(function (entry, index) {
      const position = adjustedDomainLeft - leftOffset / innerWidth;
      updates[entry.axisName + ".position"] = Math.max(0, position);
      if (index < left.length - 1) leftOffset += entry.railWidth;
    });
    let rightOffset = 0;
    right.forEach(function (entry, index) {
      const position = adjustedDomainRight + rightOffset / innerWidth;
      updates[entry.axisName + ".position"] = Math.min(1, position);
      if (index < right.length - 1) rightOffset += entry.railWidth;
    });

    const currentDomain = layout.xaxis?.domain || [0, 1];
    const inputMargin = gd.layout?.margin || {};
    let changed = !almostEqual(inputMargin.l, marginLeft, 1) ||
      !almostEqual(inputMargin.r, marginRight, 1) ||
      !almostEqual(currentDomain[0], adjustedDomainLeft) ||
      !almostEqual(currentDomain[1], adjustedDomainRight);
    entries.forEach(function (entry) {
      if (!almostEqual(entry.axis.position, updates[entry.axisName + ".position"])) changed = true;
    });
    const signature = JSON.stringify(Object.keys(updates).sort().map(function (key) {
      const value = updates[key];
      return [key, Array.isArray(value)
        ? value.map(function (item) { return Number(item).toFixed(5); })
        : Number(value).toFixed(5)];
    }));
    const repeatedImmediately = workspace._multiAxisRailSignature === signature &&
      Date.now() - (workspace._multiAxisRailAppliedAt || 0) < 400;
    if (!changed || repeatedImmediately) return false;

    workspace._multiAxisRailSignature = signature;
    workspace._multiAxisRailAppliedAt = Date.now();
    workspace.dataset.multiAxisRailRelayout = "true";
    Promise.resolve(window.Plotly.relayout(gd, updates)).catch(function () {
      delete workspace._multiAxisRailSignature;
    }).finally(function () {
      delete workspace.dataset.multiAxisRailRelayout;
    });
    return true;
  }

  function attachAxisZoneEvents(zone, workspace, gd, axisName) {
    zone.addEventListener("pointerdown", function (event) {
      if (event.button !== 0 || workspace.classList.contains("multi-axis-dnd-active")) return;
      const axis = gd._fullLayout?.[axisName];
      const startRange = linearRange(axis);
      const pixelsPerUnit = Number(axis?._m);
      if (!startRange || !Number.isFinite(pixelsPerUnit) || pixelsPerUnit === 0) return;

      event.preventDefault();
      event.stopPropagation();
      workspace.dataset.multiAxisAxisDragging = "true";
      zone.setPointerCapture?.(event.pointerId);
      zone.style.cursor = "grabbing";
      const startY = event.clientY;
      let frame = 0;
      let latestY = startY;

      function applyPan() {
        frame = 0;
        const delta = latestY - startY;
        const shifted = startRange.map(function (value) {
          return value - delta / pixelsPerUnit;
        });
        relayoutAxis(gd, axisName, dataRange(axis, shifted));
      }

      function move(moveEvent) {
        latestY = moveEvent.clientY;
        moveEvent.preventDefault();
        moveEvent.stopPropagation();
        if (!frame) frame = requestAnimationFrame(applyPan);
      }

      function finish(finishEvent) {
        if (frame) {
          cancelAnimationFrame(frame);
          applyPan();
        }
        zone.style.cursor = "ns-resize";
        delete workspace.dataset.multiAxisAxisDragging;
        zone.releasePointerCapture?.(finishEvent.pointerId);
        zone.removeEventListener("pointermove", move);
        zone.removeEventListener("pointerup", finish);
        zone.removeEventListener("pointercancel", finish);
        finishEvent.preventDefault();
        finishEvent.stopPropagation();
        requestAnimationFrame(function () { syncAxisHitZones(workspace, gd); });
      }

      zone.addEventListener("pointermove", move);
      zone.addEventListener("pointerup", finish);
      zone.addEventListener("pointercancel", finish);
    });

    zone.addEventListener("wheel", function (event) {
      const axis = gd._fullLayout?.[axisName];
      const range = linearRange(axis);
      if (!range) return;
      const rect = zone.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (rect.bottom - event.clientY) / rect.height));
      const centre = range[0] + (range[1] - range[0]) * fraction;
      const zoom = Math.exp(Math.max(-20, Math.min(20, event.deltaY)) / 200);
      const scaled = range.map(function (value) {
        return centre + (value - centre) * zoom;
      });
      event.preventDefault();
      event.stopPropagation();
      relayoutAxis(gd, axisName, dataRange(axis, scaled));
    }, {passive: false});

    zone.addEventListener("dblclick", function (event) {
      if (!window.Plotly) return;
      const update = {};
      update[axisName + ".autorange"] = true;
      event.preventDefault();
      event.stopPropagation();
      window.Plotly.relayout(gd, update);
    });
  }

  function syncAxisHitZones(workspace, gd) {
    if (!workspace?.isConnected || !gd?._fullLayout) return;
    if (workspace.dataset.multiAxisAxisDragging === "true") return;
    if (syncAxisRailLayout(workspace, gd)) return;
    let layer = workspace.querySelector(":scope > .multi-axis-axis-hit-layer");
    if (!layer) {
      layer = document.createElement("div");
      layer.className = "multi-axis-axis-hit-layer";
      workspace.appendChild(layer);
    }
    layer.replaceChildren();
    const workspaceRect = workspace.getBoundingClientRect();
    const gdRect = gd.getBoundingClientRect();
    const entries = Object.keys(gd._fullLayout).filter(function (key) {
      return /^yaxis\d*$/.test(key);
    }).map(function (axisName) {
      const axis = gd._fullLayout[axisName];
      if (!axis || axis.visible === false || axis._id === "y") return null;
      const bounds = axisTickBounds(gd, axis);
      if (!bounds) return null;
      return {
        axisName: axisName,
        axis: axis,
        bounds: bounds,
        lineX: axis.side === "right" ? bounds.left - 5 : bounds.right + 5
      };
    }).filter(Boolean);

    ["left", "right"].forEach(function (side) {
      const sideEntries = entries.filter(function (entry) {
        return entry.axis.side === side;
      }).sort(function (a, b) { return a.lineX - b.lineX; });

      sideEntries.forEach(function (entry, index) {
        const axisName = entry.axisName;
        const axis = entry.axis;
        const bounds = entry.bounds;
        const previous = sideEntries[index - 1];
        const next = sideEntries[index + 1];
        const leftEdge = previous
          ? (previous.lineX + entry.lineX) / 2
          : bounds.left - 6;
        const rightEdge = next
          ? (entry.lineX + next.lineX) / 2
          : bounds.right + 6;
      const zone = document.createElement("div");
      zone.className = "multi-axis-axis-hit-zone";
      zone.dataset.axisName = axisName;
      zone.title = "Перетащите для сдвига; колесо — масштаб оси";
        zone.style.left = Math.max(0, leftEdge - workspaceRect.left) + "px";
        zone.style.width = Math.max(14, rightEdge - leftEdge) + "px";
      zone.style.top = Math.max(0, gdRect.top - workspaceRect.top + axis._offset) + "px";
      zone.style.height = Math.max(24, axis._length) + "px";
      attachAxisZoneEvents(zone, workspace, gd, axisName);
      layer.appendChild(zone);
      });
    });
  }

  function bindAxisInteractions(workspace) {
    const gd = workspace.querySelector(".js-plotly-plot");
    if (!gd || typeof gd.on !== "function") return false;
    if (gd.dataset.multiAxisInteractionsBound === "true") return true;
    gd.dataset.multiAxisInteractionsBound = "true";
    gd.on("plotly_afterplot", function () {
      requestAnimationFrame(function () { syncAxisHitZones(workspace, gd); });
    });
    requestAnimationFrame(function () { syncAxisHitZones(workspace, gd); });
    return true;
  }

  function discoverAxisWorkspaces() {
    document.querySelectorAll(WORKSPACE).forEach(bindAxisInteractions);
  }

  const axisObserver = new MutationObserver(discoverAxisWorkspaces);
  axisObserver.observe(document.documentElement, {childList: true, subtree: true});
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", discoverAxisWorkspaces, {once: true});
  } else {
    discoverAxisWorkspaces();
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
