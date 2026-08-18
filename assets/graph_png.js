/** Shared PNG export operations for the main Plotly graph. */
(function () {
  "use strict";

  function getPlot(graphId) {
    const host = document.getElementById(graphId || "graph");
    const plot = host && host.querySelector(".js-plotly-plot");
    if (!plot || !window.Plotly) {
      throw new Error("График не найден в DOM.");
    }
    return plot;
  }

  function getVisibleSize(plot) {
    const svg = plot.querySelector("svg.main-svg") || plot.querySelector("svg");
    const rect = (svg || plot).getBoundingClientRect();
    return {
      width: Math.max(1, Math.round(rect.width)),
      height: Math.max(1, Math.round(rect.height)),
    };
  }

  function dataUrlToBlob(dataUrl) {
    const parts = dataUrl.split(",");
    const mime = (parts[0].match(/:(.*?);/) || [, "image/png"])[1];
    const binary = atob(parts[1]);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new Blob([bytes], { type: mime });
  }

  function render(graphId) {
    const plot = getPlot(graphId);
    const size = getVisibleSize(plot);
    return window.Plotly.toImage(plot, {
      format: "png",
      width: size.width,
      height: size.height,
      scale: 1,
    });
  }

  async function copyToClipboard(graphId) {
    if (
      !window.isSecureContext ||
      !window.ClipboardItem ||
      !navigator.clipboard ||
      !navigator.clipboard.write
    ) {
      throw new Error("Копирование изображений в буфер недоступно.");
    }

    const dataUrl = await render(graphId);
    const blob = dataUrlToBlob(dataUrl);
    await navigator.clipboard.write([
      new window.ClipboardItem({ [blob.type]: blob }),
    ]);
  }

  async function saveToFile(graphId) {
    const dataUrl = await render(graphId);
    const link = document.createElement("a");
    const stamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
    link.href = dataUrl;
    link.download = `graph-${stamp}.png`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  window.graphPng = Object.freeze({
    copyToClipboard,
    saveToFile,
  });
})();
