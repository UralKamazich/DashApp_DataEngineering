/** Drop local Excel/PKL files onto the dataset panel, its tab or file button. */
(function () {
  "use strict";

  const targetSelector = ".dataset-file-drop-target";

  function isFileDrag(event) {
    return Array.from(event.dataTransfer?.types || []).includes("Files");
  }

  function targets() {
    return document.querySelectorAll(targetSelector);
  }

  function clearHover() {
    targets().forEach(function (target) {
      target.classList.remove("dataset-file-drop-hover");
    });
  }

  function openDatasetPanel() {
    const drawer = document.getElementById("dataset-drawer");
    if (!drawer || drawer.classList.contains("open")) return;
    window.dash_clientside.set_props("dataset-drawer", {
      className: (drawer.className + " open").trim()
    });
    window.dash_clientside.set_props("dataset-drawer-open-state", {data: true});
  }

  document.addEventListener("dragover", function (event) {
    const target = event.target.closest(targetSelector);
    if (!target || !isFileDrag(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    if (target.id === "dataset-side-tab") openDatasetPanel();
    targets().forEach(function (item) {
      item.classList.toggle("dataset-file-drop-hover", item === target);
    });
  });

  document.addEventListener("dragleave", function (event) {
    const target = event.target.closest(targetSelector);
    if (!target) return;
    if (!event.relatedTarget || !target.contains(event.relatedTarget)) {
      target.classList.remove("dataset-file-drop-hover");
    }
  });

  document.addEventListener("drop", function (event) {
    const target = event.target.closest(targetSelector);
    if (!target || !isFileDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();

    const files = Array.from(event.dataTransfer.files || []);
    const file = files.find(function (candidate) {
      return /\.(xlsx|pkl)$/i.test(candidate.name || "");
    }) || files[0];
    const filePath = file?.path || "";

    clearHover();
    if (!file || !filePath) return;
    window.dash_clientside.set_props("dataset-file-drop-store", {
      data: {path: filePath, name: file.name, nonce: Date.now()}
    });
  });

  document.addEventListener("dragend", clearHover);
})();
