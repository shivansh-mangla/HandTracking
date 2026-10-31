const visionPackageUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
const visionWasmUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const handModelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const handConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

export class HandTrackingSession {
  constructor({ video, canvas, onState, onFrame }) {
    this.video = video;
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onState = onState;
    this.onFrame = onFrame;
    this.generation = 0;
    this.animationFrame = null;
    this.stream = null;
    this.landmarker = null;
    this.lastFrameTime = 0;
    this.smoothedFps = 0;
  }

  async start() {
    if (this.stream || this.generation && this.starting) {
      return;
    }

    const generation = ++this.generation;
    this.starting = true;
    this.onState("loading", "Loading hand tracking and starting your camera…");

    let landmarker;
    try {
      if (!window.isSecureContext) {
        throw new Error("Camera access requires HTTPS, or localhost while testing.");
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not support camera access.");
      }

      const { FilesetResolver, HandLandmarker } = await import(visionPackageUrl);
      if (generation !== this.generation) {
        return;
      }

      const vision = await FilesetResolver.forVisionTasks(visionWasmUrl);
      if (generation !== this.generation) {
        return;
      }

      landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: handModelUrl },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      if (generation !== this.generation) {
        landmarker.close();
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      });
      if (generation !== this.generation) {
        stream.getTracks().forEach((track) => track.stop());
        landmarker.close();
        return;
      }

      this.landmarker = landmarker;
      this.stream = stream;
      this.video.srcObject = stream;
      await this.video.play();
      if (generation !== this.generation) {
        return;
      }

      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      this.lastFrameTime = 0;
      this.smoothedFps = 0;
      this.onState("running", "Camera is on. Looking for a hand…");
      this.processFrame(generation);
    } catch (error) {
      if (generation !== this.generation) {
        return;
      }

      this.releaseResources();
      this.onState("error", this.describeError(error));
    } finally {
      if (generation === this.generation) {
        this.starting = false;
      }
    }
  }

  stop() {
    this.generation += 1;
    this.starting = false;
    this.releaseResources();
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.onFrame({ hands: 0, fps: 0 });
    this.onState("idle", "Camera is off. Start when you’re ready.");
  }

  processFrame(generation) {
    if (generation !== this.generation || !this.video.videoWidth) {
      return;
    }

    const now = performance.now();
    if (this.lastFrameTime) {
      const instantaneousFps = 1000 / (now - this.lastFrameTime);
      this.smoothedFps = this.smoothedFps
        ? this.smoothedFps * 0.85 + instantaneousFps * 0.15
        : instantaneousFps;
    }
    this.lastFrameTime = now;

    const result = this.landmarker.detectForVideo(this.video, now);
    this.drawLandmarks(result.landmarks ?? []);
    this.onFrame({ hands: result.landmarks?.length ?? 0, fps: Math.round(this.smoothedFps) });
    this.animationFrame = window.requestAnimationFrame(() => this.processFrame(generation));
  }

  drawLandmarks(hands) {
    const { width, height } = this.canvas;
    this.context.clearRect(0, 0, width, height);

    for (const landmarks of hands) {
      this.context.lineWidth = Math.max(2, width / 240);
      this.context.lineCap = "round";
      this.context.strokeStyle = "#d8d0ff";
      this.context.fillStyle = "#ffffff";

      for (const [start, end] of handConnections) {
        this.context.beginPath();
        this.context.moveTo(landmarks[start].x * width, landmarks[start].y * height);
        this.context.lineTo(landmarks[end].x * width, landmarks[end].y * height);
        this.context.stroke();
      }

      for (const landmark of landmarks) {
        this.context.beginPath();
        this.context.arc(
          landmark.x * width,
          landmark.y * height,
          Math.max(2.5, width / 150),
          0,
          Math.PI * 2
        );
        this.context.fill();
        this.context.strokeStyle = "#7866d6";
        this.context.lineWidth = Math.max(1.5, width / 320);
        this.context.stroke();
      }
    }
  }

  releaseResources() {
    if (this.animationFrame !== null) {
      window.cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    this.video.srcObject = null;
    this.landmarker?.close();
    this.landmarker = null;
  }

  describeError(error) {
    if (error instanceof Error && (
      error.message.startsWith("Camera access") ||
      error.message.startsWith("This browser")
    )) {
      return error.message;
    }

    switch (error?.name) {
      case "NotAllowedError":
      case "PermissionDeniedError":
        return "Camera permission was denied. Allow camera access in your browser settings, then try again.";
      case "NotFoundError":
      case "DevicesNotFoundError":
        return "No camera was found. Connect a camera and try again.";
      case "NotReadableError":
      case "TrackStartError":
        return "The camera could not be started. It may already be in use by another app.";
      case "OverconstrainedError":
        return "The camera could not provide a supported video mode.";
      default:
        return `Hand tracking could not start: ${error instanceof Error ? error.message : "an unexpected error occurred"}`;
    }
  }
}
