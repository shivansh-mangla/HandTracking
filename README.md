# Hand Studio

Hand Studio is a small web interface for hand-tracking experiments. It provides a responsive activity menu and four interactive activities:

- Hand tracking
- **Distance estimate** — estimates hand-to-camera distance from the span between two palm landmarks. The estimate is approximate and depends on hand size, camera, and lighting.
- **Drag-box game** — pinch with your thumb and index finger to pick up and move boxes.
- **Volume demo** — control an optional in-page tone with a pinch gesture or slider. Turn off gesture control to adjust the slider while the camera is running. It does not change device or server volume.

Each activity shares the same camera and hand-tracking layer. Camera access is requested only after you select **Start camera**. Video frames are processed in your browser and are not sent to a server. The first start downloads the MediaPipe hand-tracking runtime and model from their CDNs. Camera access requires HTTPS, except when testing on `localhost`. The menu does not request camera access.

## Run locally

From the repository root, start Python's static file server:

```sh
python3 -m http.server 8000
```

Then open <http://localhost:8000> in a supported browser. No frontend packages or build step are required. The Google Fonts import is optional; the page falls back to system sans-serif fonts if fonts cannot be loaded.
