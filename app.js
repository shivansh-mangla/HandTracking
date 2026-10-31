import { HandTrackingSession } from "./hand-tracking.js";

const activities = {
  "hand-tracking": {
    title: "Hand tracking",
    category: "LANDMARKS",
    description: "See your hand landmarks come to life, in real time.",
    interactive: true
  },
  "distance-estimate": {
    title: "Distance estimate",
    category: "ESTIMATION",
    description: "Explore how the apparent size of a hand can be used to make a rough estimate of its distance."
  },
  "drag-box": {
    title: "Drag-box game",
    category: "PINCH TO MOVE",
    description: "Use a pinch gesture to pick up a box and move it around the screen."
  },
  "volume-control": {
    title: "Volume demo",
    category: "GESTURE CONTROL",
    description: "Try the idea of using the distance between two fingers to control an in-page audio volume."
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
  },
  onFrame: ({ hands, fps }) => {
    handCount.textContent = String(hands);
    fpsValue.textContent = fps ? String(fps) : "—";
    trackingStatus.textContent = hands
      ? `${hands === 1 ? "One hand" : `${hands} hands`} detected. Move your hands to explore the landmarks.`
      : "Camera is on. Looking for a hand…";
  }
});

function renderRoute({ focusHeading = false } = {}) {
  const match = window.location.hash.match(/^#\/activity\/([a-z-]+)$/);
  const activity = match ? activities[match[1]] : null;

  if (!activity) {
    handTrackingSession.stop();
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
  previewNote.hidden = activity.interactive === true;
  previewArt.hidden = activity.interactive === true;
  trackingPanel.hidden = activity.interactive !== true;
  homeView.hidden = true;
  activityView.hidden = false;
  document.title = `${activity.title} — Hand Studio`;
  window.scrollTo({ top: 0, behavior: "smooth" });

  if (!activity.interactive) {
    handTrackingSession.stop();
  }

  if (focusHeading) {
    activityTitle.focus();
  }
}

window.addEventListener("hashchange", () => renderRoute({ focusHeading: true }));
startCameraButton.addEventListener("click", () => handTrackingSession.start());
stopCameraButton.addEventListener("click", () => handTrackingSession.stop());
window.addEventListener("pagehide", () => handTrackingSession.stop());
renderRoute();
