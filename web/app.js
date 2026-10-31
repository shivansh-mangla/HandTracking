import { HandTrackingSession } from "./hand-tracking.js";
import { currentActivity } from "./activities/catalog.js";
import { distanceBetween, interpolateDistance } from "./activities/measurements.js";
import { DragBoxActivity } from "./activities/drag-box.js";
import { VolumeDemo } from "./activities/volume-demo.js";

const homeView = document.querySelector("#home-view");
const activityView = document.querySelector("#activity-view");
const welcomeDialog = document.querySelector("#welcome-dialog");
const preloadStatus = document.querySelector("#preload-status");
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
let welcomeDialogShown = false;

const dragActivity = new DragBoxActivity({
  board: dragBoard,
  cursor: dragCursor,
  status: dragStatus,
  count: boxesMoved,
  canvas: landmarkCanvas
});
const volumeDemo = new VolumeDemo({
  slider: volumeSlider,
  value: volumeValue,
  status: volumeStatus,
  toggleButton: toggleToneButton
});

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
      distanceValue.textContent = String(Math.round(interpolateDistance(span, landmarkCanvas.width)));
      distanceStatus.textContent = "Estimate updates as your hand moves closer to or farther from the camera.";
    }
  } else if (activity?.mode === "drag") {
    dragActivity.update(landmarks?.[0] ?? null);
  } else if (activity?.mode === "volume") {
    if (landmarks?.length && gestureVolume.checked) {
      const pinchSpan = distanceBetween(landmarks[0][4], landmarks[0][8], landmarkCanvas.width, landmarkCanvas.height);
      volumeDemo.setVolume((pinchSpan - 18) / (180 - 18) * 100);
      volumeDemo.setStatus("Gesture detected. Pinch closer or spread your fingers to adjust the tone.");
    } else if (gestureVolume.checked) {
      volumeDemo.setStatus("Show one hand and pinch your thumb and index finger to adjust the tone.");
    } else {
      volumeDemo.setStatus("Gesture control is off. Adjust the volume with the slider.");
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
    if (state === "loading") {
      preloadStatus.textContent = "Preparing hand tracking and requesting camera access…";
    } else if (state === "running") {
      preloadStatus.textContent = "Hand tracking is ready.";
    } else if (state === "error") {
      preloadStatus.textContent = `Hand tracking could not start: ${message}`;
    }
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

void handTrackingSession.preload().then(
  () => {
    preloadStatus.textContent = "Hand tracking is ready.";
  },
  (error) => {
    const message = error instanceof Error ? error.message : "an unexpected error occurred";
    preloadStatus.textContent = `Hand tracking could not be prepared: ${message}`;
  }
);

function renderRoute({ focusHeading = false } = {}) {
  const activity = currentActivity();

  if (!activity) {
    handTrackingSession.stop();
    volumeDemo.stop();
    homeView.hidden = false;
    activityView.hidden = true;
    if (!welcomeDialogShown) {
      welcomeDialog.showModal();
      welcomeDialogShown = true;
    }
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
  if (welcomeDialog.open) {
    welcomeDialog.close();
  }
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
    dragActivity.update(null);
  }
  if (activity.mode !== "volume") {
    volumeDemo.stop();
  }

  if (focusHeading) {
    activityTitle.focus();
  }
}

window.addEventListener("hashchange", () => renderRoute({ focusHeading: true }));
window.addEventListener("pagehide", () => handTrackingSession.stop());
window.addEventListener("pagehide", () => volumeDemo.stop());
startCameraButton.addEventListener("click", () => void handTrackingSession.start());
stopCameraButton.addEventListener("click", () => handTrackingSession.stop());
resetBoxesButton.addEventListener("click", () => dragActivity.reset());

renderRoute();
