export const activities = {
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

export function currentActivity() {
  const match = window.location.hash.match(/^#\/activity\/([a-z-]+)$/);
  return match ? activities[match[1]] : null;
}
