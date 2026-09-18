/* Background Remover — upload, options, and job polling.
   No framework, no build step: this file runs as-is. */
(() => {
  "use strict";

  const POLL_INTERVAL_MS = 400;
  const TERMINAL = ["done", "error", "cancelled"];

  const state = { files: [], jobs: new Map(), capabilities: null };

  const $ = (id) => document.getElementById(id);
  const els = {};

  function setStatus(message, tone = "") {
    els.status.textContent = message;
    els.status.dataset.tone = tone;
  }

  async function loadCapabilities() {
    const response = await fetch("/api/capabilities");
    if (!response.ok) throw new Error("capabilities unavailable");
    state.capabilities = await response.json();

    const labels = {
      auto: "Auto",
      cuda: "NVIDIA GPU (CUDA)",
      mps: "Apple GPU (Metal)",
      cpu: "CPU",
    };
    els.device.innerHTML = "";
    for (const name of ["auto", ...state.capabilities.devices]) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = labels[name] || name;
      els.device.append(option);
    }

    const active = state.capabilities.default_device;
    const gpu = state.capabilities.devices.some((d) => d !== "cpu");
    els.deviceNote.textContent = gpu
      ? `A GPU is available. Auto currently selects ${active}. Choose CPU to turn it off.`
      : "No GPU detected on this machine; everything runs on the CPU.";
  }

  function syncModeOptions() {
    const mode = els.mode.value;
    for (const row of document.querySelectorAll("[data-when-mode]")) {
      row.hidden = row.dataset.whenMode !== mode;
    }
  }

  function addFiles(fileList) {
    for (const file of fileList) state.files.push(file);
    renderQueue();
    setStatus(`${state.files.length} file(s) ready.`);
  }

  function removeFile(index) {
    const job = state.jobs.get(index);
    if (job && job.objectUrl) URL.revokeObjectURL(job.objectUrl);
    state.files.splice(index, 1);
    const remapped = new Map();
    state.jobs.forEach((value, key) => {
      if (key < index) remapped.set(key, value);
      else if (key > index) remapped.set(key - 1, value);
    });
    state.jobs = remapped;
    renderQueue();
    setStatus(state.files.length ? `${state.files.length} file(s) ready.` : "");
  }

  function clearAll() {
    for (const job of state.jobs.values()) {
      if (job.objectUrl) URL.revokeObjectURL(job.objectUrl);
    }
    state.files = [];
    state.jobs.clear();
    renderQueue();
    setStatus("");
  }

  function describe(job) {
    // A queued-but-not-submitted row now carries an empty job object (it holds
    // the thumbnail's object URL), so test the status rather than the object.
    if (!job || !job.status) return "Ready";
    if (job.status === "error") return job.error || "Failed";
    if (job.status === "cancelled") return "Cancelled";
    if (job.status === "done") return `Done on ${job.device_used || "cpu"}`;
    if (job.kind === "video" && job.frames_total) {
      return `Frame ${job.frames_done} of ${job.frames_total}`;
    }
    // The server names the step it is on, which distinguishes slow from hung.
    // Before the first stage lands, say so rather than implying work started.
    if (job.stage) return job.stage;
    return job.status === "queued" ? "Waiting for a free worker" : "Starting";
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function percent(job) {
    return Math.round((job && job.progress ? job.progress : 0) * 100);
  }

  function thumbnailFor(file, job) {
    const box = document.createElement("div");
    box.className = "thumb";
    if (!file.type.startsWith("image/")) {
      box.classList.add("thumb--video");
      box.textContent = "VIDEO";
      return box;
    }
    job.objectUrl = job.objectUrl || URL.createObjectURL(file);
    const img = document.createElement("img");
    img.src = job.objectUrl;
    img.alt = "";
    img.loading = "lazy";
    // Blur-in on first decode so a slow thumbnail does not pop.
    img.addEventListener("load", () => img.classList.add("is-ready"), { once: true });
    box.append(img);
    return box;
  }

  function buildPreview(file, job) {
    const wrapper = document.createElement("div");
    wrapper.className = "queue-item__preview";
    // compare.js upgrades this into a draggable before/after slider.
    wrapper.dataset.compare = "";
    wrapper.dataset.before = job.objectUrl;
    wrapper.dataset.after = `/api/jobs/${job.id}/result`;
    wrapper.dataset.kind = job.kind || "image";
    queueMicrotask(() =>
      document.dispatchEvent(new CustomEvent("bgr:preview", { detail: { wrapper } })),
    );
    return wrapper;
  }

  function renderQueue() {
    els.queue.innerHTML = "";
    state.files.forEach((file, index) => {
      const job = state.jobs.get(index) || {};
      state.jobs.set(index, job); // keep objectUrl stable across renders
      const item = document.createElement("li");
      item.className = "queue-item";
      item.dataset.index = String(index);
      if (job.status) item.dataset.status = job.status;

      item.append(thumbnailFor(file, job));

      const main = document.createElement("div");
      main.className = "queue-item__main";

      const head = document.createElement("div");
      head.className = "queue-item__head";
      const name = document.createElement("span");
      name.className = "queue-item__name";
      name.textContent = file.name;
      const size = document.createElement("span");
      size.className = "queue-item__size";
      size.textContent = formatSize(file.size);
      head.append(name, size);

      const meta = document.createElement("div");
      meta.className = "queue-item__meta";
      meta.textContent = describe(job);

      main.append(head, meta);

      const active = job.status === "running" || job.status === "queued";
      if (active) {
        const row = document.createElement("div");
        row.className = "bar-row";
        const track = document.createElement("div");
        track.className = "bar";
        track.setAttribute("role", "progressbar");
        track.setAttribute("aria-valuemin", "0");
        track.setAttribute("aria-valuemax", "100");
        track.setAttribute("aria-valuenow", String(percent(job)));
        const fill = document.createElement("div");
        fill.className = "bar__fill";
        fill.style.width = `${percent(job)}%`;
        track.append(fill);
        const pct = document.createElement("span");
        pct.className = "bar__pct";
        pct.textContent = `${percent(job)}%`;
        row.append(track, pct);
        main.append(row);
      }

      item.append(main);

      const actions = document.createElement("div");
      actions.className = "queue-item__actions";
      if (active) {
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.className = "button button--small";
        cancel.textContent = "Cancel";
        cancel.addEventListener("click", () => cancelJob(job.id));
        actions.append(cancel);
      }
      if (job.status === "done") {
        const link = document.createElement("a");
        link.className = "button button--small";
        link.href = `/api/jobs/${job.id}/result`;
        link.textContent = "Download";
        link.download = job.result_name || "";
        actions.append(link);
      }
      if (!job.status) {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "button button--small";
        remove.textContent = "Remove";
        remove.addEventListener("click", () => removeFile(index));
        actions.append(remove);
      }
      item.append(actions);

      if (job.status === "done") item.append(buildPreview(file, job));

      els.queue.append(item);
    });

    const anyDone = [...state.jobs.values()].some((j) => j.status === "done");
    els.downloadAll.hidden = !anyDone;
    els.clear.hidden = state.files.length === 0;
  }

  function buildForm(file) {
    const form = new FormData();
    form.append("file", file);
    form.append("mode", els.mode.value);
    form.append("color", els.color.value);
    form.append("blur_radius", els.blur.value);
    form.append("device", els.device.value);
    if (els.mode.value === "image" && els.background.files[0]) {
      form.append("background", els.background.files[0]);
    }
    return form;
  }

  async function cancelJob(jobId) {
    try {
      await fetch(`/api/jobs/${jobId}`, { method: "DELETE" });
    } catch {
      /* the poll loop will report the real state */
    }
  }

  async function submit() {
    if (state.files.length === 0) {
      setStatus("Choose at least one file first.", "error");
      return;
    }
    if (els.mode.value === "image" && !els.background.files[0]) {
      setStatus("Pick a background image, or choose a different output mode.", "error");
      return;
    }

    els.start.disabled = true;
    setStatus("Uploading…");

    for (const [index, file] of state.files.entries()) {
      const existing = state.jobs.get(index);
      if (existing && existing.status === "done") continue;
      try {
        const response = await fetch("/api/jobs", { method: "POST", body: buildForm(file) });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error || `Upload failed (${response.status})`);
        state.jobs.set(index, Object.assign({}, state.jobs.get(index) || {}, body));
        renderQueue();
        await poll(index, body.id);
      } catch (error) {
        const prev = state.jobs.get(index) || {};
        state.jobs.set(index, Object.assign({}, prev, { status: "error", error: error.message }));
        renderQueue();
      }
    }

    els.start.disabled = false;
    const failed = [...state.jobs.values()].filter((j) => j.status === "error").length;
    setStatus(failed ? `${failed} file(s) failed.` : "All done.", failed ? "error" : "ok");
  }

  async function poll(index, jobId) {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      const response = await fetch(`/api/jobs/${jobId}`);
      if (!response.ok) throw new Error("Lost track of that job.");
      const body = await response.json();
      const previous = state.jobs.get(index) || {};
      state.jobs.set(index, Object.assign({}, previous, body));
      renderQueue();
      if (TERMINAL.includes(body.status)) return body;
    }
  }

  async function downloadAll() {
    const ids = [...state.jobs.values()].filter((j) => j.status === "done").map((j) => j.id);
    const response = await fetch("/api/jobs/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    if (!response.ok) {
      setStatus("Could not build the zip.", "error");
      return;
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "backgrounds.zip";
    link.click();
    URL.revokeObjectURL(url);
  }

  function wireDropZone() {
    const zone = els.dropZone;
    zone.addEventListener("click", () => els.fileInput.click());
    zone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        els.fileInput.click();
      }
    });
    for (const type of ["dragenter", "dragover"]) {
      zone.addEventListener(type, (event) => {
        event.preventDefault();
        zone.classList.add("is-dragging");
      });
    }
    for (const type of ["dragleave", "drop"]) {
      zone.addEventListener(type, () => zone.classList.remove("is-dragging"));
    }
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      addFiles(event.dataTransfer.files);
    });
  }

  function init() {
    Object.assign(els, {
      dropZone: $("drop-zone"),
      fileInput: $("file-input"),
      mode: $("mode"),
      color: $("color"),
      blur: $("blur-radius"),
      background: $("background-input"),
      device: $("device"),
      deviceNote: $("device-note"),
      queue: $("queue"),
      status: $("status-region"),
      start: $("start"),
      downloadAll: $("download-all"),
      clear: $("clear"),
    });

    wireDropZone();
    els.fileInput.addEventListener("change", (event) => addFiles(event.target.files));
    els.mode.addEventListener("change", syncModeOptions);
    els.start.addEventListener("click", submit);
    els.downloadAll.addEventListener("click", downloadAll);
    els.clear.addEventListener("click", clearAll);

    syncModeOptions();
    loadCapabilities().catch(() => setStatus("Could not reach the server.", "error"));
  }

  window.BGR = { state, addFiles, submit, poll, renderQueue, clearAll, removeFile };
  document.addEventListener("DOMContentLoaded", init);
})();
