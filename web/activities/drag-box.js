import { distanceBetween } from "./measurements.js";

const initialBoxes = [
  { x: 0.2, y: 0.28 },
  { x: 0.73, y: 0.36 },
  { x: 0.55, y: 0.76 }
];

export class DragBoxActivity {
  constructor({ board, cursor, status, count, canvas }) {
    this.board = board;
    this.cursor = cursor;
    this.status = status;
    this.count = count;
    this.canvas = canvas;
    this.elements = [...board.querySelectorAll(".drag-box")];
    this.boxes = initialBoxes.map((position) => ({ ...position }));
    this.grabbedBox = null;
  }

  reset() {
    this.boxes = initialBoxes.map((position) => ({ ...position }));
    this.grabbedBox = null;
    this.update(null);
    this.status.textContent = "Boxes reset. Pinch one to move it.";
  }

  update(landmarks) {
    if (!landmarks) {
      this.cursor.hidden = true;
      this.grabbedBox = null;
    } else {
      const thumb = landmarks[4];
      const indexFinger = landmarks[8];
      const pinchX = 1 - (thumb.x + indexFinger.x) / 2;
      const pinchY = (thumb.y + indexFinger.y) / 2;
      const pinching = distanceBetween(
        thumb,
        indexFinger,
        this.canvas.width,
        this.canvas.height
      ) / this.canvas.width < 0.075;

      this.cursor.hidden = false;
      this.cursor.style.left = `${pinchX * 100}%`;
      this.cursor.style.top = `${pinchY * 100}%`;
      this.cursor.classList.toggle("is-pinching", pinching);

      if (pinching && this.grabbedBox === null) {
        const nearestBox = this.boxes.findIndex((box) =>
          Math.abs(box.x - pinchX) < 0.14 && Math.abs(box.y - pinchY) < 0.18
        );
        if (nearestBox !== -1) {
          this.grabbedBox = nearestBox;
          this.status.textContent = "Box picked up! Keep pinching and move your hand.";
        }
      }

      if (pinching && this.grabbedBox !== null) {
        this.boxes[this.grabbedBox] = {
          x: Math.max(0.1, Math.min(0.9, pinchX)),
          y: Math.max(0.12, Math.min(0.88, pinchY))
        };
      } else if (!pinching && this.grabbedBox !== null) {
        this.grabbedBox = null;
        this.status.textContent = "Box released. Pinch another box to keep playing.";
      } else if (!pinching) {
        this.status.textContent = "Pinch a box with your thumb and index finger, then move your hand.";
      }
    }

    this.elements.forEach((element, index) => {
      element.style.left = `${this.boxes[index].x * 100}%`;
      element.style.top = `${this.boxes[index].y * 100}%`;
      element.classList.toggle("is-grabbed", this.grabbedBox === index);
    });
    this.count.textContent = String(this.boxes.filter((box, index) =>
      Math.hypot(box.x - initialBoxes[index].x, box.y - initialBoxes[index].y) > 0.08
    ).length);
  }
}
