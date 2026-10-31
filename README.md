# Hand Studio

Hand Studio is a small web interface for hand-tracking experiments. It currently provides a responsive activity menu and a live hand-landmark tracker, plus preview screens for:

- Hand tracking
- Distance estimate
- Drag-box game
- Volume demo

The distance estimate, drag-box game, and volume demo are previews; their Python demos have not yet been ported to the browser. The menu does not request camera access.

The hand-tracking screen requests camera access only after you select **Start camera**. Video frames are processed in your browser and are not sent to a server. The first start downloads the MediaPipe hand-tracking runtime and model from their CDNs. Camera access requires HTTPS, except when testing on `localhost`.

## Run locally

From the repository root, start Python's static file server:

```sh
python3 -m http.server 8000
```

Then open <http://localhost:8000> in a supported browser. No frontend packages or build step are required. The Google Fonts import is optional; the page falls back to system sans-serif fonts if fonts cannot be loaded.
