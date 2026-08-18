/** Searchable field picker opened from a GraphWorkspace drop zone. */
(function () {
  "use strict";

  let activeZone = null;
  let datasetColumns = [];
  let sortColumns = false;

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

  function getDatasetColumns() {
    const seen = new Set();
    return Array.from(
      document.querySelectorAll("#columns-badges [data-column-name]")
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
        '<div class="field-picker-order">' +
          '<span class="field-picker-order-label is-active" data-order-label="dataset">Dataset</span>' +
          '<button type="button" class="field-picker-switch" role="switch" aria-checked="false" aria-label="Сортировка столбцов">' +
            '<span class="field-picker-switch-knob"></span>' +
          '</button>' +
          '<span class="field-picker-order-label" data-order-label="sort">Sort</span>' +
        '</div>' +
      '</div>' +
      '<div class="field-picker-list" role="listbox"></div>' +
      '<footer class="field-picker-footer">ПКМ по другой зоне переключит поле</footer>';

    picker.querySelector(".field-picker-close").addEventListener("click", closePicker);
    picker.querySelector(".field-picker-search").addEventListener("input", renderColumns);
    picker.querySelector(".field-picker-switch").addEventListener("click", function () {
      sortColumns = !sortColumns;
      updateSortControl();
      renderColumns();
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
    const toggle = picker.querySelector(".field-picker-switch");
    toggle.setAttribute("aria-checked", String(sortColumns));
    picker.querySelector('[data-order-label="dataset"]').classList.toggle("is-active", !sortColumns);
    picker.querySelector('[data-order-label="sort"]').classList.toggle("is-active", sortColumns);
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

  function openPicker(zone, x, y) {
    createPicker();
    activeZone = zone;
    datasetColumns = getDatasetColumns();

    const picker = document.getElementById("graph-field-picker");
    picker.querySelector(".field-picker-title").textContent =
      zone.getAttribute("data-default-label") || "Столбец";
    picker.querySelector(".field-picker-search").value = "";
    picker.classList.add("is-open");
    document.getElementById("graph-workspace")?.classList.add("field-picker-open");
    updateSortControl();
    renderColumns();
    positionPicker(x, y);
    picker.querySelector(".field-picker-search").focus({ preventScroll: true });
  }

  function closePicker() {
    const picker = document.getElementById("graph-field-picker");
    if (picker) picker.classList.remove("is-open");
    document.getElementById("graph-workspace")?.classList.remove("field-picker-open");
    activeZone = null;
  }

  function clearZone(zone) {
    const emptyValue = zone.getAttribute("data-drop-mode") === "append" ? [] : null;
    writeZoneValue(zone, emptyValue);
    if (zone === activeZone) renderColumns();
  }

  function installListeners() {
    document.addEventListener("contextmenu", function (event) {
      const zone = event.target.closest(".graph-drop-zone");
      if (!zone) return;
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

    document.addEventListener("dragstart", closePicker);
    window.addEventListener("resize", closePicker);
  }

  function init() {
    createPicker();
    installListeners();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
