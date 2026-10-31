import { HandTrackingSession } from "./hand-tracking.js";

const activities = {
  "hand-tracking": {
    title: "Hand tracking",
    category: "LANDMARKS",
    description: "See your hand landmarks come to life, in real time.",
    mode: "tracking"
  },
  "distance-estimate": {
    title: "Distance estimate",
    category: "ESTIMATION",
    description: "Show one hand to see a rough estimate based on its apparent size.",
    mode: "distance"
  },
  "drag-box": {
    title: "Drag-box game",
    category: "PINCH TO MOVE",
    description: "Pinch a box with your thumb and index finger, then move your hand to drag it.",
    mode: "drag"
  },
  "volume-control": {
    title: "Volume demo",
    category: "GESTURE CONTROL",
    description: "Use a pinch gesture to adjust a gentle tone generated right here in the page.",
    mode: "volume"
  }
};

const homeView = document.querySelector("#home-view");
const activityView = document.querySelector("#activity-view");
const activityTitle = document.querySelector("#activity-title");
const activityCategory = document.querySelector("#activity-category");
const activityDescription = document.querySelector("#activity-description");
const previewNote = document.querySelector("#preview-note");
const previewArt = document.querySelector("#preview-art");
const trackingPanel = document.querySelector("#tracking-panel");
const cameraVideo = document.querySelector("#camera-video");
const landmarkCanvas = document.querySelector("#landmark-canvas");
const trackingStatus = document.querySelector("#tracking-status");
const trackingError = document.querySelector("#tracking-error");
const startCameraButton = document.querySelector("#start-camera");
const stopCameraButton = document.querySelector("#stop-camera");
const handCount = document.querySelector("#hand-count");
const fpsValue = document.querySelector("#fps-value");
const distanceTools = document.querySelector("#distance-tools");
const distanceValue = document.querySelector("#distance-value");
const distanceStatus = document.querySelector("#distance-status");
const dragTools = document.querySelector("#drag-tools");
const dragBoard = document.querySelector("#drag-board");
const dragCursor = document.querySelector("#drag-cursor");
const dragStatus = document.querySelector("#drag-status");
const boxesMoved = document.querySelector("#boxes-moved");
const resetBoxesButton = document.querySelector("#reset-boxes");
const volumeTools = document.querySelector("#volume-tools");
const volumeSlider = document.querySelector("#volume-slider");
const volumeValue = document.querySelector("#volume-value");
const volumeStatus = document.querySelector("#volume-status");
const gestureVolume = document.querySelector("#gesture-volume");
const toggleToneButton = document.querySelector("#toggle-tone");

const calibration = {
  pixelSpans: [57, 59, 62, 67, 70, 75, 80, 87, 93, 103, 112, 130, 145, 170, 200, 245, 300],
  centimeters: [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20]
};
const initialBoxes = [
  { x: 0.2, y: 0.28 },
  { x: 0.73, y: 0.36 },
  { x: 0.55, y: 0.76 }
];
let boxes = initialBoxes.map((position) => ({ ...position }));
let grabbedBox = null;
let audioContext = null;
let audioOscillator = null;
let audioGain = null;

function currentActivity() {
  const match = window.location.hash.match(/^#\/activity\/([a-z-]+)$/);
  return match ? activities[match[1]] : null;
}

function interpolateDistance(pixelSpan) {
  const referenceSpan = pixelSpan * 640 / landmarkCanvas.width;
  const { pixelSpans, centimeters } = calibration;
  if (referenceSpan <= pixelSpans[0]) {
    return centimeters[0];
  }
  if (referenceSpan >= pixelSpans[pixelSpans.length - 1]) {
    return centimeters[centimeters.length - 1];
  }

  const upperIndex = pixelSpans.findIndex((span) => span >= referenceSpan);
  const lowerIndex = upperIndex - 1;
  const progress = (referenceSpan - pixelSpans[lowerIndex]) /
    (pixelSpans[upperIndex] - pixelSpans[lowerIndex]);
  return centimeters[lowerIndex] +
    progress * (centimeters[upperIndex] - centimeters[lowerIndex]);
}

function distanceBetween(first, second, width, height) {
  return Math.hypot((first.x - second.x) * width, (first.y - second.y) * height);
}

function setVolume(value) {
  const volume = Math.max(0, Math.min(100, Math.round(value)));
  volumeSlider.value = String(volume);
  volumeValue.textContent = String(volume);
  if (audioGain && audioContext) {
    audioGain.gain.setTargetAtTime(volume / 100 * 0.18, audioContext.currentTime, 0.04);
  }
}

function updateDragBoxes(landmarks) {
  const elements = [...dragBoard.querySelectorAll(".drag-box")];
  if (!landmarks) {
    dragCursor.hidden = true;
    grabbedBox = null;
  } else {
    const thumb = landmarks[4];
    const indexFinger = landmarks[8];
    const pinchX = 1 - (thumb.x + indexFinger.x) / 2;
    const pinchY = (thumb.y + indexFinger.y) / 2;
    const pinching = distanceBetween(thumb, indexFinger, landmarkCanvas.width, landmarkCanvas.height) /
      landmarkCanvas.width < 0.075;

    dragCursor.hidden = false;
    dragCursor.style.left = `${pinchX * 100}%`;
    dragCursor.style.top = `${pinchY * 100}%`;
    dragCursor.classList.toggle("is-pinching", pinching);

    if (pinching && grabbedBox === null) {
      const nearestBox = boxes.findIndex((box) =>
        Math.abs(box.x - pinchX) < 0.14 && Math.abs(box.y - pinchY) < 0.18
      );
      if (nearestBox !== -1) {
        grabbedBox = nearestBox;
        dragStatus.textContent = "Box picked up! Keep pinching and move your hand.";
      }
    }

    if (pinching && grabbedBox !== null) {
      boxes[grabbedBox] = {
        x: Math.max(0.1, Math.min(0.9, pinchX)),
        y: Math.max(0.12, Math.min(0.88, pinchY))
      };
    } else if (!pinching && grabbedBox !== null) {
      grabbedBox = null;
      dragStatus.textContent = "Box released. Pinch another box to keep playing.";
    } else if (!pinching) {
      dragStatus.textContent = "Pinch a box with your thumb and index finger, then move your hand.";
    }
  }

  elements.forEach((element, index) => {
    element.style.left = `${boxes[index].x * 100}%`;
    element.style.top = `${boxes[index].y * 100}%`;
    element.classList.toggle("is-grabbed", grabbedBox === index);
  });
  boxesMoved.textContent = String(boxes.filter((box, index) =>
    Math.hypot(box.x - initialBoxes[index].x, box.y - initialBoxes[index].y) > 0.08
  ).length);
}

function updateActivityFrame({ hands, fps, landmarks }) {
  handCount.textContent = String(hands);
  fpsValue.textContent = fps ? String(fps) : "—";
  const activity = currentActivity();

  if (activity?.mode === "tracking") {
    trackingStatus.textContent = hands
      ? `${hands === 1 ? "One hand" : `${hands} hands`} detected. Move your hands to explore the landmarks.`
      : "Camera is on. Looking for a hand…";
  } else if (activity?.mode === "distance") {
    if (!landmarks?.length) {
      distanceValue.textContent = "—";
      distanceStatus.textContent = "Show one hand to see an estimate.";
    } else {
      const span = distanceBetween(landmarks[0][5], landmarks[0][17], landmarkCanvas.width, landmarkCanvas.height);
      distanceValue.textContent = String(Math.round(interpolateDistance(span)));
      distanceStatus.textContent = "Estimate updates as your hand moves closer to or farther from the camera.";
    }
  } else if (activity?.mode === "drag") {
    updateDragBoxes(landmarks?.[0] ?? null);
  } else if (activity?.mode === "volume") {
    if (landmarks?.length && gestureVolume.checked) {
      const pinchSpan = distanceBetween(landmarks[0][4], landmarks[0][8], landmarkCanvas.width, landmarkCanvas.height);
      setVolume((pinchSpan - 18) / (180 - 18) * 100);
      volumeStatus.textContent = "Gesture detected. Pinch closer or spread your fingers to adjust the tone.";
    } else if (gestureVolume.checked) {
      volumeStatus.textContent = "Show one hand and pinch your thumb and index finger to adjust the tone.";
    } else {
      volumeStatus.textContent = "Gesture control is off. Adjust the volume with the slider.";
    }
  }
}

const handTrackingSession = new HandTrackingSession({
  video: cameraVideo,
  canvas: landmarkCanvas,
  onState: (state, message) => {
    trackingStatus.textContent = state === "error" ? "Camera could not start." : message;
    trackingError.hidden = state !== "error";
    trackingError.textContent = state === "error" ? message : "";
    startCameraButton.hidden = state === "running";
    startCameraButton.disabled = state === "loading";
    startCameraButton.textContent = state === "loading" ? "Preparing camera…" : "Start camera →";
    stopCameraButton.hidden = state !== "running";

    if (state === "idle" || state === "error") {
      const prefix = state === "error" ? "Camera unavailable. " : "";
      distanceValue.textContent = "—";
      distanceStatus.textContent = `${prefix}Start the camera and show one hand to see an estimate.`;
      dragStatus.textContent = `${prefix}Start the camera, pinch a box with your thumb and index finger, then move your hand.`;
      if (gestureVolume.checked) {
        volumeStatus.textContent = `${prefix}Start the camera and pinch your thumb and index finger to adjust the tone.`;
      }
    }
  },
  onFrame: updateActivityFrame
});

function renderRoute({ focusHeading = false } = {}) {
  const activity = currentActivity();

  if (!activity) {
    handTrackingSession.stop();
    stopTone();
    homeView.hidden = false;
    activityView.hidden = true;
    document.title = "Hand Studio — choose an activity";
    window.scrollTo({ top: 0, behavior: "smooth" });

    if (focusHeading) {
      document.querySelector("#page-title").focus();
    }

    if (window.location.hash && window.location.hash !== "#/") {
      window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}#/`);
    }

    return;
  }

  activityTitle.textContent = activity.title;
  activityCategory.textContent = activity.category;
  activityDescription.textContent = activity.description;
  previewNote.hidden = true;
  previewArt.hidden = true;
  trackingPanel.hidden = false;
  distanceTools.hidden = activity.mode !== "distance";
  dragTools.hidden = activity.mode !== "drag";
  dragBoard.hidden = activity.mode !== "drag";
  volumeTools.hidden = activity.mode !== "volume";
  homeView.hidden = true;
  activityView.hidden = false;
  document.title = `${activity.title} — Hand Studio`;
  window.scrollTo({ top: 0, behavior: "smooth" });

  if (activity.mode !== "drag") {
    updateDragBoxes(null);
  }
  if (activity.mode !== "volume") {
    stopTone();
  }

  if (focusHeading) {
    activityTitle.focus();
  }
}

function stopTone() {
  if (audioOscillator) {
    audioOscillator.stop();
    audioOscillator.disconnect();
    audioOscillator = null;
  }
  audioGain?.disconnect();
  audioGain = null;

  if (audioContext) {
    void audioContext.close();
    audioContext = null;
  }
  toggleToneButton.textContent = "Enable demo sound";
  if (volumeStatus) {
    volumeStatus.textContent = "Sound is off. Enable it to hear the gesture-controlled tone.";
  }
}

window.addEventListener("hashchange", () => renderRoute({ focusHeading: true }));
window.addEventListener("pagehide", () => handTrackingSession.stop());
window.addEventListener("pagehide", stopTone);
startCameraButton.addEventListener("click", () => handTrackingSession.start());
stopCameraButton.addEventListener("click", () => handTrackingSession.stop());
resetBoxesButton.addEventListener("click", () => {
  boxes = initialBoxes.map((position) => ({ ...position }));
  grabbedBox = null;
  updateDragBoxes(null);
  dragStatus.textContent = "Boxes reset. Pinch one to move it.";
});
volumeSlider.addEventListener("input", () => setVolume(volumeSlider.value));
toggleToneButton.addEventListener("click", async () => {
  if (audioContext) {
    stopTone();
    return;
  }

  try {
    const AudioContextConstructor = window.AudioContext ?? window.webkitAudioContext;
    if (!AudioContextConstructor) {
      throw new Error("This browser does not support Web Audio.");
    }

    const context = new AudioContextConstructor();
    audioContext = context;
    await context.resume();
    if (audioContext !== context) {
      return;
    }

    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = 440;
    gain.gain.value = Number(volumeSlider.value) / 100 * 0.18;
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    audioOscillator = oscillator;
    audioGain = gain;
    toggleToneButton.textContent = "Stop demo sound";
    volumeStatus.textContent = "Tone is playing. Use your hand or the slider to change its volume.";
  } catch (error) {
    stopTone();
    volumeStatus.textContent = error instanceof Error
      ? error.message
      : "The demo sound could not be started.";
  }
});

renderRoute();
