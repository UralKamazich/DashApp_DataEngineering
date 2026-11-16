/**
 * «Экспресс корр. анализ»: rating table + floating cross-plot windows.
 *
 * The table is plain DOM (fast for thousands of channels).  Clicking a row
 * opens a draggable window to the right of the cursor with a canvas scatter
 * plot; windows stay open until closed with ×, any number can be opened, and
 * each can copy its image to the clipboard.
 */
(function () {
  "use strict";

  const METHOD_LABELS = { pearson: "Пирсон", spearman: "Спирмен" };
  const state = { result: null, sortKey: "abs_r", asc: false, filter: "" };
  const openedCount = new Map();
  let zIndex = 1500;
  let windowSeq = 0;

  // ---------------------------------------------------------------- helpers
  function esc(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function fmtInt(value) {
    return Number(value || 0).toLocaleString("ru-RU");
  }

  function fmtR(value) {
    return value == null || !isFinite(value) ? "—" : Number(value).toFixed(4);
  }

  function fmtP(value) {
    if (value == null || !isFinite(value)) return "—";
    if (value === 0) return "< 1e-300";
    return value < 1e-4 ? Number(value).toExponential(2) : Number(value).toFixed(4);
  }

  function fmtAxis(value, step) {
    if (!isFinite(value)) return "";
    const abs = Math.abs(value);
    if (abs !== 0 && (abs >= 1e6 || abs < 1e-4)) return value.toExponential(2);
    const decimals = Math.min(8, Math.max(0, -Math.floor(Math.log10(step || 1)) + 1));
    return Number(value.toFixed(decimals)).toString();
  }

  // ------------------------------------------------------------------ table
  function sortedRows() {
    const result = state.result;
    if (!result || !Array.isArray(result.rows)) return [];
    const filter = state.filter.trim().toLowerCase();
    const rows = result.rows
      .map((row, index) => ({ row, index }))
      .filter((item) => !filter || item.row.channel.toLowerCase().includes(filter));
    const key = state.sortKey;
    const dir = state.asc ? 1 : -1;
    rows.sort((a, b) => {
      if (key === "rank") return (a.index - b.index) * dir;
      if (key === "channel") return a.row.channel.localeCompare(b.row.channel, "ru") * dir;
      const av = a.row[key], bv = b.row[key];
      const an = av == null ? -Infinity : av, bn = bv == null ? -Infinity : bv;
      return (an - bn) * dir;
    });
    return rows;
  }

  function draw() {
    const host = document.getElementById("xcorr-table");
    if (!host) return;
    const result = state.result;
    if (!result) {
      host.innerHTML = '<div class="xcorr-empty">Выберите канал и нажмите «Рассчитать».</div>';
      return;
    }
    if (result.error) {
      host.innerHTML = '<div class="xcorr-empty">' + esc(result.error) + "</div>";
      return;
    }
    const rows = sortedRows();
    const head = [
      ["rank", "№", "num"], ["channel", "Канал", ""], ["r", "r", "num"],
      ["abs_r", "|r|", ""], ["n", "N", "num"], ["p", "p-value", "num"],
    ].map(([key, label, cls]) => {
      const sorted = state.sortKey === key;
      const classes = [cls, sorted ? "is-sorted" : "", sorted && state.asc ? "is-asc" : ""]
        .filter(Boolean).join(" ");
      return '<th data-key="' + key + '" class="' + classes + '">' + label + "</th>";
    }).join("");

    const body = rows.map(({ row, index }) => {
      const sign = row.r >= 0 ? "pos" : "neg";
      const width = Math.round(Math.min(1, Math.abs(row.r)) * 100);
      const opened = openedCount.get(row.channel) ? ' class="is-opened"' : "";
      return (
        '<tr data-index="' + index + '"' + opened + ">" +
        '<td class="rank num">' + (index + 1) + "</td>" +
        '<td class="channel" title="' + esc(row.channel) + '">' + esc(row.channel) + "</td>" +
        '<td class="num xcorr-r-' + sign + '">' + fmtR(row.r) + "</td>" +
        '<td><div class="xcorr-bar"><span class="' + sign + '" style="width:' + width + '%"></span></div></td>' +
        '<td class="num">' + fmtInt(row.n) + "</td>" +
        '<td class="num">' + fmtP(row.p) + "</td>" +
        "</tr>"
      );
    }).join("");

    host.innerHTML = rows.length
      ? '<table class="xcorr-table"><thead><tr>' + head + "</tr></thead><tbody>" + body + "</tbody></table>"
      : '<div class="xcorr-empty">Нет каналов, подходящих под фильтр.</div>';
  }

  function render(result) {
    state.result = result || null;
    draw();
  }

  document.addEventListener("click", function (event) {
    const header = event.target.closest && event.target.closest("#xcorr-table thead th");
    if (header) {
      const key = header.getAttribute("data-key");
      if (state.sortKey === key) {
        state.asc = !state.asc;
      } else {
        state.sortKey = key;
        state.asc = key === "channel" || key === "rank" || key === "p";
      }
      draw();
      return;
    }
    const row = event.target.closest && event.target.closest("#xcorr-table tbody tr");
    if (row && state.result) {
      const item = state.result.rows[Number(row.getAttribute("data-index"))];
      if (item) openWindow(state.result, item, event.clientX, event.clientY);
    }
  });

  document.addEventListener("input", function (event) {
    const target = event.target;
    if (target && (target.id === "xcorr-filter" || target.closest?.("#xcorr-filter"))) {
      state.filter = target.value || "";
      draw();
    }
  });

  // ------------------------------------------------------------ scatter plot
  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const unit = raw / power;
    const nice = unit < 1.5 ? 1 : unit < 3 ? 2 : unit < 7 ? 5 : 10;
    return nice * power;
  }

  function extent(values) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!isFinite(min)) return [0, 1];
    if (min === max) return [min - 1, max + 1];
    const pad = (max - min) * 0.03;
    return [min - pad, max + pad];
  }

  function drawScatter(plot) {
    const { canvas, data } = plot;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const m = { left: 66, right: 14, top: 24, bottom: 42 };
    const pw = Math.max(10, width - m.left - m.right);
    const ph = Math.max(10, height - m.top - m.bottom);
    const [x0, x1] = extent(data.x);
    const [y0, y1] = extent(data.y);
    const sx = (v) => m.left + ((v - x0) / (x1 - x0)) * pw;
    const sy = (v) => m.top + ph - ((v - y0) / (y1 - y0)) * ph;
    plot.view = { m, pw, ph, x0, x1, y0, y1 };

    // grid + ticks
    ctx.font = "11px -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.strokeStyle = "#eef0f2";
    ctx.fillStyle = "#495057";
    ctx.lineWidth = 1;
    const xStep = niceStep(x1 - x0, Math.max(2, Math.floor(pw / 80)));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let v = Math.ceil(x0 / xStep) * xStep; v <= x1; v += xStep) {
      const px = Math.round(sx(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(px, m.top); ctx.lineTo(px, m.top + ph); ctx.stroke();
      ctx.fillText(fmtAxis(v, xStep), px, m.top + ph + 4);
    }
    const yStep = niceStep(y1 - y0, Math.max(2, Math.floor(ph / 50)));
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = Math.ceil(y0 / yStep) * yStep; v <= y1; v += yStep) {
      const py = Math.round(sy(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(m.left, py); ctx.lineTo(m.left + pw, py); ctx.stroke();
      ctx.fillText(fmtAxis(v, yStep), m.left - 5, py);
    }
    ctx.strokeStyle = "#adb5bd";
    ctx.strokeRect(m.left + 0.5, m.top + 0.5, pw, ph);

    // points
    ctx.save();
    ctx.beginPath(); ctx.rect(m.left, m.top, pw, ph); ctx.clip();
    const n = data.x.length;
    const size = n > 20000 ? 2 : n > 3000 ? 3 : 4;
    ctx.fillStyle = n > 5000 ? "rgba(28,126,214,0.35)" : "rgba(28,126,214,0.6)";
    for (let i = 0; i < n; i++) {
      ctx.fillRect(sx(data.x[i]) - size / 2, sy(data.y[i]) - size / 2, size, size);
    }
    if (data.fit) {
      ctx.strokeStyle = "#e03131";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(sx(data.fit.x[0]), sy(data.fit.y[0]));
      ctx.lineTo(sx(data.fit.x[1]), sy(data.fit.y[1]));
      ctx.stroke();
    }
    ctx.restore();

    // titles (kept inside the image so a copied picture is self-describing)
    ctx.fillStyle = "#212529";
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.font = "600 12px -apple-system, Segoe UI, Roboto, sans-serif";
    const title = data.y_column + "  vs  " + data.x_column;
    ctx.fillText(title, m.left, 16, pw);
    ctx.font = "11px -apple-system, Segoe UI, Roboto, sans-serif";
    ctx.textAlign = "right";
    ctx.fillStyle = "#495057";
    ctx.fillText("r = " + fmtR(data.pearson) + "   ρ = " + fmtR(data.spearman), width - m.right, 16);
    ctx.textAlign = "center";
    ctx.fillText(data.x_column, m.left + pw / 2, height - 8, pw);
    ctx.save();
    ctx.translate(13, m.top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(data.y_column, 0, 0, ph);
    ctx.restore();
  }

  // ---------------------------------------------------------------- windows
  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function placeWindow(win, cx, cy) {
    const w = win.offsetWidth, h = win.offsetHeight;
    let left = cx + 18;
    if (left + w > window.innerWidth - 8) left = cx - w - 18;
    left = clamp(left, 8, Math.max(8, window.innerWidth - w - 8));
    const top = clamp(cy - 40, 8, Math.max(8, window.innerHeight - h - 8));
    win.style.left = left + "px";
    win.style.top = top + "px";
  }

  function setNote(win, text, isError) {
    const note = win.querySelector(".xcorr-window-note");
    note.textContent = text;
    note.classList.toggle("is-error", Boolean(isError));
    clearTimeout(note._timer);
    note._timer = setTimeout(() => { note.textContent = ""; }, isError ? 4000 : 1600);
  }

  async function copyPlot(win, plot) {
    try {
      if (!plot.data) throw new Error("график ещё не построен");
      if (!window.isSecureContext || !navigator.clipboard?.write || !window.ClipboardItem) {
        throw new Error("буфер обмена недоступен");
      }
      const blob = await new Promise((resolve) => plot.canvas.toBlob(resolve, "image/png"));
      await navigator.clipboard.write([new window.ClipboardItem({ "image/png": blob })]);
      setNote(win, "Скопировано");
    } catch (error) {
      setNote(win, "Не скопировано: " + (error?.message || error), true);
    }
  }

  function markOpened(channel, delta) {
    const count = (openedCount.get(channel) || 0) + delta;
    if (count > 0) openedCount.set(channel, count); else openedCount.delete(channel);
    document.querySelectorAll("#xcorr-table tbody tr").forEach((tr) => {
      const item = state.result?.rows[Number(tr.getAttribute("data-index"))];
      if (item && item.channel === channel) tr.classList.toggle("is-opened", count > 0);
    });
  }

  function openWindow(result, row, cx, cy) {
    windowSeq += 1;
    const win = document.createElement("div");
    win.className = "xcorr-window";
    win.style.zIndex = String(++zIndex);
    win.innerHTML =
      '<div class="xcorr-window-head">' +
      '<div class="xcorr-window-title"></div>' +
      '<span class="xcorr-window-note"></span>' +
      '<button type="button" class="xcorr-window-btn xcorr-window-copy" title="Копировать изображение в буфер обмена">⧉ Копировать</button>' +
      '<button type="button" class="xcorr-window-btn xcorr-window-close" title="Закрыть" aria-label="Закрыть">×</button>' +
      "</div>" +
      '<div class="xcorr-window-body"><canvas></canvas><div class="xcorr-window-loading">Загрузка…</div></div>' +
      '<div class="xcorr-window-foot">&nbsp;</div>';
    const title = result.target + " ↔ " + row.channel;
    win.querySelector(".xcorr-window-title").textContent = title;
    win.querySelector(".xcorr-window-title").title = title;
    document.body.appendChild(win);
    placeWindow(win, cx, cy);
    markOpened(row.channel, +1);

    const plot = { canvas: win.querySelector("canvas"), data: null, view: null, stats: "" };
    const foot = win.querySelector(".xcorr-window-foot");
    const loading = win.querySelector(".xcorr-window-loading");
    const observer = new ResizeObserver(() => { if (plot.data) drawScatter(plot); });
    observer.observe(win.querySelector(".xcorr-window-body"));

    win.addEventListener("pointerdown", () => { win.style.zIndex = String(++zIndex); });
    win.querySelector(".xcorr-window-close").addEventListener("click", () => {
      observer.disconnect();
      win.remove();
      markOpened(row.channel, -1);
    });
    win.querySelector(".xcorr-window-copy").addEventListener("click", () => copyPlot(win, plot));

    // drag by the header
    const head = win.querySelector(".xcorr-window-head");
    head.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest("button")) return;
      const start = { x: event.clientX, y: event.clientY, left: win.offsetLeft, top: win.offsetTop };
      head.setPointerCapture(event.pointerId);
      win.classList.add("is-dragging");
      const move = (e) => {
        win.style.left = clamp(start.left + e.clientX - start.x, 40 - win.offsetWidth, window.innerWidth - 40) + "px";
        win.style.top = clamp(start.top + e.clientY - start.y, 0, window.innerHeight - 30) + "px";
      };
      const up = () => {
        win.classList.remove("is-dragging");
        head.removeEventListener("pointermove", move);
        head.removeEventListener("pointerup", up);
        head.removeEventListener("pointercancel", up);
      };
      head.addEventListener("pointermove", move);
      head.addEventListener("pointerup", up);
      head.addEventListener("pointercancel", up);
    });

    // cursor readout
    plot.canvas.addEventListener("mousemove", (event) => {
      const v = plot.view;
      if (!v) return;
      const rect = plot.canvas.getBoundingClientRect();
      const px = event.clientX - rect.left, py = event.clientY - rect.top;
      if (px < v.m.left || px > v.m.left + v.pw || py < v.m.top || py > v.m.top + v.ph) {
        foot.textContent = plot.stats;
        return;
      }
      const x = v.x0 + ((px - v.m.left) / v.pw) * (v.x1 - v.x0);
      const y = v.y0 + ((v.m.top + v.ph - py) / v.ph) * (v.y1 - v.y0);
      foot.textContent = "x = " + fmtAxis(x, (v.x1 - v.x0) / 1000) + "   y = " + fmtAxis(y, (v.y1 - v.y0) / 1000);
    });
    plot.canvas.addEventListener("mouseleave", () => { foot.textContent = plot.stats; });

    const params = new URLSearchParams({
      ref: result.ref || "",
      x: row.channel,
      y: result.target,
      zero: result.exclude_zero ? "1" : "0",
    });
    fetch("/api/express-corr/crossplot?" + params.toString())
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || response.statusText);
        return payload;
      })
      .then((data) => {
        plot.data = data;
        loading.remove();
        plot.stats =
          "Пирсон r = " + fmtR(data.pearson) + " · Спирмен ρ = " + fmtR(data.spearman) +
          " · N = " + fmtInt(data.total) +
          (data.shown < data.total ? " · показано " + fmtInt(data.shown) : "") +
          (result.exclude_zero ? " · без нулей" : "");
        foot.textContent = plot.stats;
        drawScatter(plot);
      })
      .catch((error) => {
        loading.textContent = "Ошибка: " + (error?.message || error);
      });
  }

  window.expressCorr = Object.freeze({ render, openWindow, methodLabels: METHOD_LABELS });
})();
