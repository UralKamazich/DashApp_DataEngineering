(function () {
  "use strict";

  const MENU_ID = "dataset-context-menu";
  const EXPORT_ID = "dataset-context-export";
  const REQUEST_ID = "dataset-export-request";
  let selectedDatasetId = null;

  function menu() {
    return document.getElementById(MENU_ID);
  }

  function closeMenu() {
    const node = menu();
    if (!node) return;
    node.classList.remove("is-open");
    node.setAttribute("aria-hidden", "true");
    selectedDatasetId = null;
  }

  function datasetIdFromTab(tab) {
    if (tab.id === "dataset-side-tab") return "source";
    try {
      const parsed = JSON.parse(tab.id || "{}");
      if (parsed.type === "dataset-rail-tab") return String(parsed.index || "");
    } catch (_) {
      return "";
    }
    return "";
  }

  function openMenu(event, tab) {
    const node = menu();
    const datasetId = datasetIdFromTab(tab);
    if (!node || !datasetId) return;
    event.preventDefault();
    event.stopPropagation();
    selectedDatasetId = datasetId;
    node.classList.add("is-open");
    node.setAttribute("aria-hidden", "false");
    const width = 178;
    const height = 38;
    node.style.left = Math.max(6, Math.min(event.clientX + 8, window.innerWidth - width - 6)) + "px";
    node.style.top = Math.max(6, Math.min(event.clientY + 6, window.innerHeight - height - 6)) + "px";
  }

  function publishExportRequest() {
    if (!selectedDatasetId) return;
    const input = document.getElementById(REQUEST_ID);
    if (!input) return;
    const value = JSON.stringify({ dataset_id: selectedDatasetId, nonce: Date.now() });
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    closeMenu();
  }

  document.addEventListener("contextmenu", function (event) {
    const tab = event.target.closest(".dataset-rail-tab, #dataset-side-tab");
    if (!tab) {
      closeMenu();
      return;
    }
    openMenu(event, tab);
  }, true);

  document.addEventListener("click", function (event) {
    if (event.target.closest("#" + EXPORT_ID)) {
      event.preventDefault();
      publishExportRequest();
      return;
    }
    if (!event.target.closest("#" + MENU_ID)) closeMenu();
  }, true);

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closeMenu();
  });
  window.addEventListener("blur", closeMenu);
  window.addEventListener("resize", closeMenu);
  window.addEventListener("scroll", closeMenu, true);
})();
