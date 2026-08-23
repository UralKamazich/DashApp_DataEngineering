/** Movable, non-modal help window owned by an individual GraphWorkspace. */
(function () {
  "use strict";

  const OPEN_CLASS = "is-open";
  let dragState = null;

  function helpWindowFor(workspace) {
    const windowId = workspace?.getAttribute("data-help-window-id");
    return windowId ? document.getElementById(windowId) : null;
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(value, maximum));
  }

  function position(helpWindow, x, y) {
    const edge = 8;
    const gap = 12;
    const width = helpWindow.offsetWidth;
    const height = helpWindow.offsetHeight;
    let left = Number(x) + gap;
    let top = Number(y) + gap;

    if (!Number.isFinite(left)) left = edge;
    if (!Number.isFinite(top)) top = edge;
    if (left + width > window.innerWidth - edge) {
      left = Number(x) - width - gap;
    }
    left = clamp(left, edge, Math.max(edge, window.innerWidth - width - edge));
    top = clamp(top, edge, Math.max(edge, window.innerHeight - height - edge));
    helpWindow.style.left = Math.round(left) + "px";
    helpWindow.style.top = Math.round(top) + "px";
  }

  function openWindow(helpWindow, x, y) {
    if (!helpWindow) return false;
    helpWindow.classList.add(OPEN_CLASS);
    helpWindow.setAttribute("aria-hidden", "false");
    position(helpWindow, x, y);
    return true;
  }

  function open(workspace, x, y) {
    return openWindow(helpWindowFor(workspace), x, y);
  }

  function close(helpWindow) {
    if (!helpWindow) return;
    helpWindow.classList.remove(OPEN_CLASS, "is-dragging");
    helpWindow.setAttribute("aria-hidden", "true");
    if (dragState?.helpWindow === helpWindow) dragState = null;
  }

  document.addEventListener("click", function (event) {
    const trigger = event.target.closest("[data-help-window-target]");
    if (trigger) {
      const targetId = trigger.getAttribute("data-help-window-target");
      const helpWindow = targetId ? document.getElementById(targetId) : null;
      if (!helpWindow) return;
      event.preventDefault();
      event.stopPropagation();
      openWindow(helpWindow, event.clientX, event.clientY);
      return;
    }

    const closeButton = event.target.closest(".graph-help-window-close");
    if (!closeButton) return;
    event.preventDefault();
    event.stopPropagation();
    close(closeButton.closest(".graph-help-window"));
  });

  document.addEventListener("pointerdown", function (event) {
    const header = event.target.closest(".graph-help-window-header");
    const helpWindow = header?.closest(".graph-help-window.is-open");
    if (!helpWindow || event.button !== 0) return;
    if (event.target.closest("button, input, select, textarea, a, [role='button']")) return;

    const rect = helpWindow.getBoundingClientRect();
    dragState = {
      helpWindow: helpWindow,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      left: rect.left,
      top: rect.top,
    };
    helpWindow.classList.add("is-dragging");
    event.preventDefault();
  });

  document.addEventListener("pointermove", function (event) {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    const edge = 8;
    const helpWindow = dragState.helpWindow;
    const left = clamp(
      dragState.left + event.clientX - dragState.startX,
      edge,
      Math.max(edge, window.innerWidth - helpWindow.offsetWidth - edge)
    );
    const top = clamp(
      dragState.top + event.clientY - dragState.startY,
      edge,
      Math.max(edge, window.innerHeight - helpWindow.offsetHeight - edge)
    );
    helpWindow.style.left = Math.round(left) + "px";
    helpWindow.style.top = Math.round(top) + "px";
    event.preventDefault();
  });

  function endDrag(event) {
    if (!dragState || (event && event.pointerId !== dragState.pointerId)) return;
    dragState.helpWindow.classList.remove("is-dragging");
    dragState = null;
  }

  document.addEventListener("pointerup", endDrag);
  document.addEventListener("pointercancel", endDrag);
  window.addEventListener("blur", function () { endDrag(); });

  window.addEventListener("resize", function () {
    document.querySelectorAll(".graph-help-window.is-open").forEach(function (helpWindow) {
      const rect = helpWindow.getBoundingClientRect();
      position(helpWindow, rect.left - 12, rect.top - 12);
    });
  });

  window.graphHelpWindow = Object.freeze({open: open, openWindow: openWindow, close: close});
})();
