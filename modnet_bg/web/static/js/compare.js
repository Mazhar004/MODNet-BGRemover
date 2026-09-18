/* Draggable before/after divider over a result.
   Listens for app.js's bgr:preview event and upgrades the node in place. */
(() => {
  "use strict";

  function mediaElement(source, kind) {
    if (kind === "video") {
      const video = document.createElement("video");
      video.src = source;
      video.controls = false;
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.play().catch(() => {});
      return video;
    }
    const image = document.createElement("img");
    image.src = source;
    image.alt = "";
    return image;
  }

  function mount(wrapper) {
    const { before, after, kind } = wrapper.dataset;
    if (!before || !after) return;

    wrapper.classList.add("compare", "checker");
    wrapper.innerHTML = "";

    const afterLayer = document.createElement("div");
    afterLayer.className = "compare__layer compare__layer--after";
    afterLayer.append(mediaElement(after, kind));

    const beforeLayer = document.createElement("div");
    beforeLayer.className = "compare__layer compare__layer--before";
    beforeLayer.append(mediaElement(before, kind));

    const handle = document.createElement("div");
    handle.className = "compare__handle";
    handle.setAttribute("role", "slider");
    handle.setAttribute("tabindex", "0");
    handle.setAttribute("aria-label", "Reveal the original image");
    handle.setAttribute("aria-valuemin", "0");
    handle.setAttribute("aria-valuemax", "100");

    let position = 50;

    function apply(next) {
      position = Math.min(100, Math.max(0, next));
      beforeLayer.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
      handle.style.left = `${position}%`;
      handle.setAttribute("aria-valuenow", String(Math.round(position)));
    }

    function fromPointer(event) {
      const box = wrapper.getBoundingClientRect();
      apply(((event.clientX - box.left) / box.width) * 100);
    }

    let dragging = false;
    wrapper.addEventListener("pointerdown", (event) => {
      dragging = true;
      wrapper.setPointerCapture(event.pointerId);
      fromPointer(event);
    });
    wrapper.addEventListener("pointermove", (event) => dragging && fromPointer(event));
    wrapper.addEventListener("pointerup", () => {
      dragging = false;
    });
    wrapper.addEventListener("pointercancel", () => {
      dragging = false;
    });

    handle.addEventListener("keydown", (event) => {
      const step = event.shiftKey ? 10 : 2;
      if (event.key === "ArrowLeft") {
        apply(position - step);
        event.preventDefault();
      }
      if (event.key === "ArrowRight") {
        apply(position + step);
        event.preventDefault();
      }
      if (event.key === "Home") {
        apply(0);
        event.preventDefault();
      }
      if (event.key === "End") {
        apply(100);
        event.preventDefault();
      }
    });

    wrapper.append(afterLayer, beforeLayer, handle);
    apply(50);
  }

  document.addEventListener("bgr:preview", (event) => mount(event.detail.wrapper));
  window.BGRCompare = { mount };
})();
