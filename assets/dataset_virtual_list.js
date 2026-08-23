/** Window the dataset channel list without changing its drag-and-drop contract. */
(function () {
  "use strict";

  const ROW_HEIGHT = 29;
  const OVERSCAN = 8;
  let lastStart = -1;
  let lastSize = -1;
  let frame = 0;

  function virtualizationEnabled(sidebar) {
    const explicitState = sidebar.getAttribute("data-virtualization-enabled");
    if (explicitState !== null) return explicitState === "true";

    const control = document.getElementById("dataset-virtualize-columns");
    if (!control) return false;
    const checkbox = control.matches?.('input[type="checkbox"]')
      ? control
      : control.querySelector?.('input[type="checkbox"]');
    return Boolean(checkbox?.checked);
  }

  function publishWindow() {
    frame = 0;
    const sidebar = document.getElementById("columns-sidebar");
    if (!sidebar || !window.dash_clientside?.set_props) return;
    if (!virtualizationEnabled(sidebar)) {
      lastStart = -1;
      lastSize = -1;
      return;
    }

    const visibleRows = Math.max(1, Math.ceil(sidebar.clientHeight / ROW_HEIGHT));
    const size = Math.max(32, visibleRows + OVERSCAN * 2);
    const start = Math.max(0, Math.floor(sidebar.scrollTop / ROW_HEIGHT) - OVERSCAN);
    if (start === lastStart && size === lastSize) return;

    lastStart = start;
    lastSize = size;
    window.dash_clientside.set_props("dataset-virtual-window", {
      data: { start: start, size: size }
    });
  }

  function scheduleWindow() {
    if (frame) return;
    frame = window.requestAnimationFrame(publishWindow);
  }

  document.addEventListener("scroll", function (event) {
    if (event.target?.id === "columns-sidebar") scheduleWindow();
  }, true);
  window.addEventListener("resize", scheduleWindow);

  const observer = new MutationObserver(function () {
    if (document.getElementById("columns-sidebar")) scheduleWindow();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true });
  window.setTimeout(scheduleWindow, 0);
})();
