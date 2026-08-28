/** Instance-scoped graph settings popovers. */
(function () {
  "use strict";

  const OPEN_CLASS = "is-open";
  const COMMON_CLASS = "graph-settings-popover--common";
  const SPECIFIC_CLASS = "graph-settings-popover--specific";
  let dragState = null;

  function popupFor(workspace) {
    if (!workspace) return null;
    const popupId = workspace.getAttribute("data-settings-popup-id");
    return popupId ? document.getElementById(popupId) : null;
  }

  function close(popup) {
    if (!popup) return;
    popup.classList.remove(OPEN_CLASS, COMMON_CLASS, SPECIFIC_CLASS, "is-dragging");
    popup.setAttribute("aria-hidden", "true");
    if (dragState && dragState.popup === popup) dragState = null;
  }

  function closeOthers(except) {
    document.querySelectorAll(".graph-settings-popover.is-open").forEach(function (popup) {
      if (popup !== except) close(popup);
    });
  }

  function position(popup, x, y) {
    const edge = 8;
    const gap = 12;
    const width = popup.offsetWidth;
    const height = popup.offsetHeight;
    let left = Number(x) + gap;
    let top = Number(y) - 12;

    if (!Number.isFinite(left)) left = edge;
    if (!Number.isFinite(top)) top = edge;
    if (left + width > window.innerWidth - edge) {
      left = Number(x) - width - gap;
    }
    left = Math.max(edge, Math.min(left, window.innerWidth - width - edge));
    top = Math.max(edge, Math.min(top, window.innerHeight - height - edge));
    popup.style.left = Math.round(left) + "px";
    popup.style.top = Math.round(top) + "px";
  }

  function open(workspace, mode, x, y) {
    const popup = popupFor(workspace);
    if (!popup) return false;

    closeOthers(popup);
    popup.classList.remove(COMMON_CLASS, SPECIFIC_CLASS);
    popup.classList.add(OPEN_CLASS, mode === "specific" ? SPECIFIC_CLASS : COMMON_CLASS);
    popup.setAttribute("aria-hidden", "false");
    position(popup, x, y);
    return true;
  }

  function shouldCloseOutside(popup) {
    const checkboxId = popup.getAttribute("data-close-on-outside-id");
    const checkboxRoot = checkboxId && document.getElementById(checkboxId);
    if (!checkboxRoot) return false;
    const input = checkboxRoot.matches("input")
      ? checkboxRoot
      : checkboxRoot.querySelector('input[type="checkbox"]');
    return Boolean(input && input.checked);
  }

  function ownsPortalTarget(popup, target) {
    if (!popup || !target || !target.closest) return false;

    // Mantine Select renders its dropdown in a body-level portal, outside the
    // settings DOM subtree. Match that listbox back to the control through
    // aria-controls so a choice is not mistaken for an outside click.
    const portal = target.closest(".mantine-Popover-dropdown");
    if (!portal) return false;
    const listbox = target.closest("[role='listbox']") ||
      portal.querySelector("[role='listbox']");
    if (!listbox || !listbox.id) return false;
    return Boolean(
      popup.querySelector("[aria-controls='" + CSS.escape(listbox.id) + "']")
    );
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(value, maximum));
  }

  document.addEventListener("pointerdown", function (event) {
    const header = event.target.closest(".graph-settings-popover-header");
    const popup = header && header.closest(
      ".graph-settings-popover--common.is-open, .multi-axis-settings-popover.is-open"
    );
    if (!popup || event.button !== 0) return;
    if (event.target.closest("button, input, select, textarea, a, [role='button']")) return;

    const rect = popup.getBoundingClientRect();
    dragState = {
      popup: popup,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      left: rect.left,
      top: rect.top,
    };
    popup.classList.add("is-dragging");
    if (header.setPointerCapture) {
      try {
        header.setPointerCapture(event.pointerId);
      } catch (_error) {
        // Document-level pointer listeners still keep the drag working.
      }
    }
    event.preventDefault();
  });

  document.addEventListener("pointermove", function (event) {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    const edge = 8;
    const popup = dragState.popup;
    const left = clamp(
      dragState.left + event.clientX - dragState.startX,
      edge,
      Math.max(edge, window.innerWidth - popup.offsetWidth - edge)
    );
    const top = clamp(
      dragState.top + event.clientY - dragState.startY,
      edge,
      Math.max(edge, window.innerHeight - popup.offsetHeight - edge)
    );
    popup.style.left = Math.round(left) + "px";
    popup.style.top = Math.round(top) + "px";
    event.preventDefault();
  });

  function endDrag(event) {
    if (!dragState || (event && event.pointerId !== dragState.pointerId)) return;
    dragState.popup.classList.remove("is-dragging");
    dragState = null;
  }

  document.addEventListener("pointerup", endDrag);
  document.addEventListener("pointercancel", endDrag);
  window.addEventListener("blur", function () { endDrag(); });

  document.addEventListener("click", function (event) {
    const closeButton = event.target.closest(".graph-settings-popover-close");
    if (closeButton) {
      close(closeButton.closest(".graph-settings-popover"));
      return;
    }

    const workspace = event.target.closest(".graph-workspace");
    if (!workspace) return;
    const triggerId = workspace.getAttribute("data-action-open-specific-settings");
    const trigger = triggerId && event.target.closest("#" + CSS.escape(triggerId));
    if (!trigger) return;

    const rect = trigger.getBoundingClientRect();
    open(workspace, "specific", rect.right, rect.top + rect.height / 2);
  });

  document.addEventListener("mousedown", function (event) {
    document.querySelectorAll(".graph-settings-popover.is-open").forEach(function (popup) {
      const belongsToPopup = popup.contains(event.target) ||
        ownsPortalTarget(popup, event.target);
      if (!belongsToPopup && shouldCloseOutside(popup)) close(popup);
    });
  }, true);

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") return;
    document.querySelectorAll(".graph-settings-popover.is-open").forEach(close);
  });

  window.addEventListener("resize", function () {
    document.querySelectorAll(".graph-settings-popover.is-open").forEach(close);
  });

  window.graphSettingsPopover = Object.freeze({open: open, close: close});
})();
