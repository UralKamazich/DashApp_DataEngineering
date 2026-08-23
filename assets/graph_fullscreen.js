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

  function fitPlotsToHost(host) {
    if (!host) return;
    const token = (host.__graphFullscreenFitToken || 0) + 1;
    host.__graphFullscreenFitToken = token;
    const resize = function () {
      if (host.__graphFullscreenFitToken !== token || activeHost() !== host) return;
      if (!window.Plotly?.relayout) return;
      host.querySelectorAll(".js-plotly-plot").forEach(function (plot) {
        const wrapper = plot.closest(".graph-fullscreen-plot") || plot.parentElement;
        const bounds = wrapper?.getBoundingClientRect?.();
        if (!bounds?.width || !bounds?.height) return;
        try {
          window.Plotly.relayout(plot, {
            width: Math.round(bounds.width),
            height: Math.round(bounds.height),
            autosize: false,
          });
        } catch (error) {
          console.warn("[graph-fullscreen] fit failed:", error);
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
      const wrapper = plot.closest(".graph-fullscreen-plot") || plot.parentElement;
      const plotBounds = plot.getBoundingClientRect();
      const wrapperBounds = wrapper?.getBoundingClientRect?.();
      return {
        plot: plot,
        wrapper: wrapper,
        height: plot.layout?.height ?? Math.round(plotBounds.height),
        width: plot.layout?.width ?? null,
        autosize: plot.layout?.autosize ?? null,
        wrapperHeight: Math.round(wrapperBounds?.height || plotBounds.height),
        wrapperStyleHeight: wrapper?.style?.height || "",
      };
    });
  }

  function restorePlotSizes(host) {
    const states = host?.__graphFullscreenPlotSizes || [];
    if (host) host.__graphFullscreenFitToken = (host.__graphFullscreenFitToken || 0) + 1;
    if (host) delete host.__graphFullscreenPlotSizes;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        states.forEach(function (state) {
          if (!window.Plotly?.relayout || !state.plot?.isConnected) return;
          if (state.wrapper?.isConnected && state.wrapperHeight) {
            state.wrapper.style.height = state.wrapperHeight + "px";
          }
          const update = {
            height: state.height,
            width: state.width,
            autosize: state.autosize ?? state.width === null,
          };
          try {
            Promise.resolve(window.Plotly.relayout(state.plot, update)).finally(function () {
              if (state.wrapper?.isConnected) {
                state.wrapper.style.height = state.wrapperStyleHeight;
              }
            });
          } catch (error) {
            if (state.wrapper?.isConnected) {
              state.wrapper.style.height = state.wrapperStyleHeight;
            }
            console.warn("[graph-fullscreen] restore failed:", error);
          }
        });
      });
    });
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
      fitPlotsToHost(host);
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
    if (current) fitPlotsToHost(host);
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
