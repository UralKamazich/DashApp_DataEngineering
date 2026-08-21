/** Fullscreen toggle for an individual graph paper. */
(function () {
  "use strict";

  const HOST = ".graph-fullscreen-host";
  const BUTTON = ".graph-fullscreen-toggle";
  const FALLBACK = "graph-fullscreen-fallback";
  let lastHost = null;

  function activeHost() {
    return document.fullscreenElement?.closest?.(HOST) ||
      document.querySelector(HOST + "." + FALLBACK);
  }

  function resizePlots(host) {
    if (!host) return;
    const resize = function () {
      window.dispatchEvent(new Event("resize"));
      if (!window.Plotly?.Plots?.resize) return;
      host.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
        try {
          window.Plotly.Plots.resize(plot);
        } catch (error) {
          console.warn("[graph-fullscreen] resize failed:", error);
        }
      });
    };
    requestAnimationFrame(function () {
      requestAnimationFrame(resize);
    });
    setTimeout(resize, 180);
  }

  function capturePlotSizes(host) {
    if (!host) return;
    host.__graphFullscreenPlotSizes = Array.from(
      host.querySelectorAll(".js-plotly-plot")
    ).map(function (plot) {
      return {
        plot: plot,
        height: plot.layout?.height ?? null,
        width: plot.layout?.width ?? null,
      };
    });
  }

  function restorePlotSizes(host) {
    const states = host?.__graphFullscreenPlotSizes || [];
    states.forEach(function (state) {
      if (!window.Plotly?.relayout || !state.plot?.isConnected) return;
      const update = {
        height: state.height,
        width: state.width,
      };
      try {
        window.Plotly.relayout(state.plot, update);
      } catch (error) {
        console.warn("[graph-fullscreen] restore failed:", error);
      }
    });
    if (host) delete host.__graphFullscreenPlotSizes;
  }

  function syncButtons() {
    const current = activeHost();
    document.querySelectorAll(BUTTON).forEach(function (button) {
      const expanded = button.closest(HOST) === current;
      const title = expanded
        ? "Вернуть график в рабочую область"
        : "Развернуть график на весь экран";
      button.setAttribute("aria-pressed", String(expanded));
      button.setAttribute("aria-label", title);
      button.setAttribute("data-title", title);
      button.title = title;
    });
  }

  function leaveFallback(host) {
    host?.classList.remove(FALLBACK);
    document.documentElement.classList.remove("graph-fullscreen-active");
    restorePlotSizes(host);
    syncButtons();
    resizePlots(host);
    lastHost = null;
  }

  async function toggle(host) {
    if (!host) return;

    if (host.classList.contains(FALLBACK)) {
      leaveFallback(host);
      return;
    }

    try {
      if (document.fullscreenElement === host) {
        await document.exitFullscreen();
        return;
      }
      if (document.fullscreenElement) await document.exitFullscreen();
      if (typeof host.requestFullscreen !== "function") throw new Error("unsupported");
      lastHost = host;
      capturePlotSizes(host);
      await host.requestFullscreen({navigationUI: "hide"});
    } catch (_error) {
      lastHost = host;
      capturePlotSizes(host);
      host.classList.add(FALLBACK);
      document.documentElement.classList.add("graph-fullscreen-active");
      syncButtons();
      resizePlots(host);
    }
  }

  document.addEventListener("click", function (event) {
    const button = event.target.closest(BUTTON);
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    toggle(button.closest(HOST));
  });

  document.addEventListener("fullscreenchange", function () {
    const current = activeHost();
    const host = current || lastHost;
    if (!current) restorePlotSizes(host);
    syncButtons();
    resizePlots(host);
    if (!current) lastHost = null;
  });

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") return;
    const host = document.querySelector(HOST + "." + FALLBACK);
    if (host) leaveFallback(host);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", syncButtons);
  } else {
    syncButtons();
  }

  window.graphFullscreen = Object.freeze({toggle: toggle});
})();
