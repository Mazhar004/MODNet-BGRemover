/* Live camera matting. One frame in flight at a time: the next frame is
   captured only after the previous mask arrives, so a slow server drops
   frames rather than building a backlog. */
(() => {
  "use strict";

  const SEND_WIDTH = 512;

  let socket = null;
  let stream = null;
  let video = null;
  let inFlight = false;
  let running = false;
  let lastFrameAt = 0;
  let smoothedFps = 0;

  const $ = (id) => document.getElementById(id);

  function captureFrame() {
    const scale = SEND_WIDTH / video.videoWidth;
    const canvas = document.createElement("canvas");
    canvas.width = SEND_WIDTH;
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  function composite(target, frame, maskImage) {
    const ctx = target.getContext("2d");
    target.width = frame.width;
    target.height = frame.height;

    const scratch = document.createElement("canvas");
    scratch.width = frame.width;
    scratch.height = frame.height;
    const scratchCtx = scratch.getContext("2d");
    scratchCtx.drawImage(frame, 0, 0);

    const mask = document.createElement("canvas");
    mask.width = frame.width;
    mask.height = frame.height;
    mask.getContext("2d").drawImage(maskImage, 0, 0, frame.width, frame.height);

    const pixels = scratchCtx.getImageData(0, 0, frame.width, frame.height);
    const maskPixels = mask.getContext("2d").getImageData(0, 0, frame.width, frame.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      pixels.data[i + 3] = maskPixels.data[i]; // red channel of a grayscale PNG
    }

    ctx.clearRect(0, 0, target.width, target.height);
    ctx.putImageData(pixels, 0, 0);
  }

  function updateFps() {
    const now = performance.now();
    if (lastFrameAt) {
      const instant = 1000 / (now - lastFrameAt);
      smoothedFps = smoothedFps ? smoothedFps * 0.8 + instant * 0.2 : instant;
      $("webcam-fps").textContent = `${smoothedFps.toFixed(1)} frames per second`;
    }
    lastFrameAt = now;
  }

  function pump() {
    if (!running || inFlight || !socket || socket.readyState !== WebSocket.OPEN) return;
    const frame = captureFrame();
    inFlight = true;
    frame.toBlob((blob) => blob && socket.send(blob), "image/jpeg", 0.8);
  }

  async function start() {
    const canvas = $("webcam-canvas");
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280 } });
    } catch {
      $("webcam-fps").textContent = "Camera access was refused.";
      return;
    }

    video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();

    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws/realtime`);
    socket.binaryType = "blob";

    socket.addEventListener("open", () => {
      const device = $("device") ? $("device").value : "auto";
      socket.send(JSON.stringify({ device, max_edge: SEND_WIDTH }));
      running = true;
      pump();
    });

    socket.addEventListener("message", async (event) => {
      if (typeof event.data === "string") {
        const payload = JSON.parse(event.data);
        if (payload.error) $("webcam-fps").textContent = payload.error;
        inFlight = false;
        pump();
        return;
      }
      const mask = await createImageBitmap(event.data);
      composite(canvas, captureFrame(), mask);
      updateFps();
      inFlight = false;
      pump();
    });

    socket.addEventListener("close", stop);

    $("webcam-start").disabled = true;
    $("webcam-stop").disabled = false;
  }

  function stop() {
    running = false;
    inFlight = false;
    lastFrameAt = 0;
    smoothedFps = 0;

    if (stream) {
      // Release the camera. Anything less leaves the indicator light on.
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();
    socket = null;

    $("webcam-start").disabled = false;
    $("webcam-stop").disabled = true;
    $("webcam-fps").textContent = "";
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      $("webcam-panel").hidden = true;
      return;
    }
    $("webcam-start").addEventListener("click", start);
    $("webcam-stop").addEventListener("click", stop);
    window.addEventListener("beforeunload", stop);
  });
})();
