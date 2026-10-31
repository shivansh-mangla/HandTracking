# Hand Studio

Hand Studio is a collection of small hand-tracking experiments. It includes a static browser app and separate Python camera demos. The browser app can be hosted on a homelab with Docker; hand tracking runs on the visitor's device rather than on the server.

## Browser app

The web app offers four activities:

- **Hand tracking** — view hand landmarks and a live hand count and frame rate.
- **Distance estimate** — estimate hand-to-camera distance from the apparent span of two palm landmarks. This is a rough estimate, not a measurement.
- **Drag-box game** — pinch with your thumb and index finger to pick up and move boxes.
- **Volume demo** — adjust a gentle tone generated in the browser with a pinch or slider. It does not control device or server volume.

### Requirements

- A modern browser with JavaScript modules, WebAssembly, and camera support.
- A camera and permission to use it for the activities that need tracking.
- A secure page context: use HTTPS when hosting on a network. Browsers generally allow camera access on `localhost` without HTTPS.

The page can be viewed without a camera, and the menu does not request camera access. As soon as the app loads, it starts downloading and preparing the local MediaPipe runtime and model in the background, so they are ready by the time you start an activity. This is about 17 MB of browser-side assets on the first visit; your browser can cache them for later visits. No camera permission is requested until you select **Start camera**. No camera, GPU, Python, Node.js, or frontend build step is needed on the web server.

### Run locally

From the repository root, serve the `web` directory:

```sh
python3 -m http.server 8000 --directory web
```

Open <http://localhost:8000>. To use the camera from another device on your network, serve the site over HTTPS; the `localhost` exception does not apply to a LAN IP address.

### Host with Docker Compose

Install Docker Engine and the Docker Compose plugin, then run from the repository root:

```sh
docker compose up -d --build
```

The included [compose.yaml](./compose.yaml) publishes the app at `127.0.0.1:8085` on the Docker host. This is intended for a reverse proxy running on that same host. Configure the proxy with a hostname and HTTPS certificate, and set its upstream to `http://127.0.0.1:8085`. The browser must use the HTTPS hostname to grant camera access.

Useful Compose commands:

```sh
docker compose logs -f       # Follow app logs
docker compose ps            # Check container status
docker compose down          # Stop and remove the container
docker compose up -d --build # Rebuild and restart after changing files
```

To run without Compose, build and publish the container port:

```sh
docker build -t hand-studio .
docker run -d --name hand-studio --restart unless-stopped -p 8080:80 hand-studio
```

Then open <http://localhost:8080>. For a homelab deployment, expose it through an HTTPS reverse proxy rather than relying on plain HTTP from other devices.

### Accuracy and privacy

Tracking and interaction results are approximate. Detection quality and responsiveness can vary with camera quality, device performance, lighting, distance, hand size, and hand position. The distance activity is especially approximate because it estimates distance from apparent hand size and does not calibrate to each person or camera. This app is for experimentation, not precision measurement.

The camera is requested only after selecting **Start camera**. Camera frames are processed in the browser and are not uploaded to the app's server. The server delivers the static app and its bundled MediaPipe files; it does not need a camera or run inference. The page imports Google Fonts for styling; if those fonts are unavailable, it falls back to system sans-serif fonts.

## Python camera experiments

The Python demos are independent of the browser app and require a local camera and a graphical desktop session for their OpenCV windows.

Install the shared dependencies from the `python` directory:

```sh
cd python
python3 -m pip install -r requirements.txt
```

Run a demo as a module from the same directory:

```sh
python3 -m experiments.tracking
python3 -m experiments.distance
python3 -m experiments.drag_box
```

Press **q** in a demo window to quit. The volume-control experiment changes the Windows system volume and is separate from the browser's in-page tone. It requires Windows and the optional `pycaw` dependency:

```powershell
python -m pip install -r requirements-windows.txt
python -m experiments.volume_control
```

The distance estimate in the Python demo is also approximate and depends on its calibration, camera, hand size, and setup.

## Project layout

```text
web/
  activities/       Browser activity logic
  styles/           Browser styles
  vendor/mediapipe/ Locally served MediaPipe runtime, WASM, model, and license
  app.js            Browser app and activity routing
  hand-tracking.js  Camera access and hand landmark tracking
python/
  experiments/      Standalone OpenCV camera demos
  hand_tracking/    Shared Python hand detector
  requirements.txt  Shared Python dependencies
  requirements-windows.txt
                    Optional Windows volume-control dependency
Dockerfile          Static Nginx image for the browser app
compose.yaml        Localhost-bound Compose deployment
```

## Third-party assets and license

The project is licensed under the MIT License; see [LICENSE](./LICENSE). The browser app bundles MediaPipe Tasks Vision version 0.10.21 and its hand-landmarker model under `web/vendor/mediapipe/`. The MediaPipe Apache-2.0 license is included alongside those assets.
