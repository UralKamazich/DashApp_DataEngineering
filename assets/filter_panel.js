/** Drag dataset channels into the global filter panel. */
(function () {
  "use strict";

  let draggedChannel = null;

  function targets() {
    return document.querySelectorAll(".filter-drop-target");
  }

  function setDragging(active) {
    targets().forEach(function (target) {
      target.classList.toggle("filter-dnd-active", active);
      if (!active) target.classList.remove("filter-drop-hover");
    });
  }

  function openFiltersPanel() {
    const drawer = document.getElementById("filters-drawer");
    if (!drawer || drawer.classList.contains("open")) return;
    window.dash_clientside.set_props("filters-drawer", {
      className: (drawer.className + " open").trim()
    });
    window.dash_clientside.set_props("filters-drawer-open-state", {data: true});
  }

  document.addEventListener("dragstart", function (event) {
    const badge = event.target.closest("[data-column-name]");
    if (!badge) return;
    draggedChannel = badge.getAttribute("data-column-name");
    if (!draggedChannel) return;
    event.dataTransfer.setData("text/plain", draggedChannel);
    setDragging(true);
  });

  document.addEventListener("dragover", function (event) {
    const target = event.target.closest(".filter-drop-target");
    if (!target || !draggedChannel) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    if (target.id === "filters-side-tab") openFiltersPanel();
    targets().forEach(function (item) {
      item.classList.toggle("filter-drop-hover", item === target);
    });
  });

  document.addEventListener("dragleave", function (event) {
    const target = event.target.closest(".filter-drop-target");
    if (!target) return;
    if (!event.relatedTarget || !target.contains(event.relatedTarget)) {
      target.classList.remove("filter-drop-hover");
    }
  });

  document.addEventListener("drop", function (event) {
    const target = event.target.closest(".filter-drop-target");
    if (!target || !draggedChannel) return;
    event.preventDefault();
    event.stopPropagation();
    window.dash_clientside.set_props("filter-drop-store", {
      data: { column: draggedChannel, nonce: Date.now() }
    });
    draggedChannel = null;
    setDragging(false);
  });

  document.addEventListener("dragend", function () {
    draggedChannel = null;
    setDragging(false);
  });
})();
