# Hand Studio

Hand Studio is a small web interface for hand-tracking experiments. It provides a responsive activity menu and four interactive activities:

- Hand tracking
- **Distance estimate** — estimates hand-to-camera distance from the span between two palm landmarks. The estimate is approximate and depends on hand size, camera, and lighting.
- **Drag-box game** — pinch with your thumb and index finger to pick up and move boxes.
- **Volume demo** — control an optional in-page tone with a pinch gesture or slider. Turn off gesture control to adjust the slider while the camera is running. It does not change device or server volume.

Each activity shares the same camera and hand-tracking layer. Camera access is requested only after you select **Start camera**. Video frames are processed in your browser and are not sent to a server. Results are approximate and can vary with the camera, device performance, lighting, distance, and hand position. Camera access requires HTTPS, except when testing on `localhost`. The menu does not request camera access.

The MediaPipe Tasks Vision runtime, WebAssembly files, and hand model are bundled under `web/vendor/mediapipe/`, so the app does not fetch them from third-party CDNs at runtime. The runtime package is version 0.10.21; its Apache-2.0 license is included with the vendored files.

## Run the web app

From the repository root, serve the `web` directory:

```sh
python3 -m http.server 8000 --directory web
```

Then open <http://localhost:8000> in a supported browser. The browser app is organized under `web/`; shared camera tracking is in `web/hand-tracking.js`, and activity-specific logic is under `web/activities/`. No frontend packages or build step are required. The Google Fonts import is optional; the page falls back to system sans-serif fonts if fonts cannot be loaded.

## Run with Docker

Start the app using the included Compose configuration:

```sh
docker compose up -d --build
```

The Compose file binds the app to `127.0.0.1:8085`, intended for a reverse proxy running on the same host. For a standalone container, build and run with `docker build -t hand-studio .` and `docker run -d --name hand-studio --restart unless-stopped -p 8080:80 hand-studio`, then open <http://localhost:8080>. For homelab access, use a reverse proxy that provides HTTPS. Browser camera access requires HTTPS except on `localhost`, and each visitor must grant camera permission in their browser. The camera and hand tracking run on the visitor's device; the server does not need a camera.

## Run the Python experiments

The Python camera demos are independent from the web app and live under `python/`. Install their cross-platform dependencies:

```sh
python3 -m pip install -r python/requirements.txt
```

Run an experiment as a module from the `python` directory:

```sh
cd python
python3 -m experiments.tracking
```

Other modules are `experiments.distance` and `experiments.drag_box`. Press **q** in the camera window to exit. The `experiments.volume_control` demo uses Windows system-volume APIs and has separate dependencies; on Windows, install them with `python -m pip install -r requirements-windows.txt` from `python/`.
