/* Draggable before/after divider over a result.
   Listens for app.js's bgr:preview event and upgrades the node in place. */
(() => {
  "use strict";

  /* Match the frame to the media's own aspect ratio.

     Without this the wrapper keeps a fixed height and object-fit letterboxes
     the result, so most of the checkerboard is dead space rather than actual
     transparency -- which makes it hard to judge the cut-out. */
  function adoptAspectRatio(wrapper, element) {
    const apply = (w, h) => {
      if (w > 0 && h > 0) wrapper.style.aspectRatio = `${w} / ${h}`;
    };
    if (element.tagName === "VIDEO") {
      element.addEventListener(
        "loadedmetadata",
        () => apply(element.videoWidth, element.videoHeight),
        { once: true },
      );
    } else {
      const ready = () => apply(element.naturalWidth, element.naturalHeight);
      if (element.complete && element.naturalWidth) ready();
      else element.addEventListener("load", ready, { once: true });
    }
  }

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
    const afterMedia = mediaElement(after, kind);
    adoptAspectRatio(wrapper, afterMedia);
    afterLayer.append(afterMedia);

    const beforeLayer = document.createElement("div");
    beforeLayer.className = "compare__layer compare__layer--before";
    beforeLayer.append(mediaElement(before, kind));

    // Corner tags: the split is meaningless without naming the two sides.
    const beforeTag = document.createElement("span");
    beforeTag.className = "compare__tag compare__tag--before";
    beforeTag.textContent = "Original";
    const afterTag = document.createElement("span");
    afterTag.className = "compare__tag compare__tag--after";
    afterTag.textContent = "Removed";

    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "compare__handle";
    handle.setAttribute("role", "slider");
    handle.setAttribute("aria-label", "Reveal the original image");
    handle.setAttribute("aria-valuemin", "0");
    handle.setAttribute("aria-valuemax", "100");

    // Chevrons inside the puck, so it reads as draggable rather than decorative.
    const grip = document.createElement("span");
    grip.className = "compare__grip";
    grip.setAttribute("aria-hidden", "true");
    grip.innerHTML =
      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" ' +
      'stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>' +
      '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" ' +
      'stroke-linejoin="round" style="margin-left:-8px"><path d="m9 18 6-6-6-6"/></svg>';
    handle.append(grip);

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

    wrapper.append(afterLayer, beforeLayer, beforeTag, afterTag, handle);
    apply(50);
  }

  document.addEventListener("bgr:preview", (event) => mount(event.detail.wrapper));
  window.BGRCompare = { mount };
})();
