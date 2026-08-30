/** Use Electron's native file dialog while retaining the browser fallback. */
(function () {
  "use strict";

  document.addEventListener("click", async function (event) {
    const button = event.target.closest("#pick-file-btn");
    const desktop = window.dataAnalizeDesktop;
    if (!button || !desktop || typeof desktop.pickDataset !== "function") return;

    // Stop Dash from also launching the Python/tkinter fallback.
    event.preventDefault();
    event.stopImmediatePropagation();

    try {
      const selected = await desktop.pickDataset();
      if (!selected || !selected.path) return;
      window.dash_clientside.set_props("dataset-file-drop-store", {
        data: {
          path: selected.path,
          name: selected.name || selected.path.split(/[\\/]/).pop(),
          nonce: Date.now()
        }
      });
    } catch (error) {
      console.error("Native dataset dialog failed:", error);
    }
  }, true);
})();
