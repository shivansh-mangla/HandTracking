const calibration = {
  pixelSpans: [57, 59, 62, 67, 70, 75, 80, 87, 93, 103, 112, 130, 145, 170, 200, 245, 300],
  centimeters: [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20]
};

export function interpolateDistance(pixelSpan, canvasWidth) {
  const referenceSpan = pixelSpan * 640 / canvasWidth;
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

export function distanceBetween(first, second, width, height) {
  return Math.hypot((first.x - second.x) * width, (first.y - second.y) * height);
}
