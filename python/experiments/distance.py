import cv2
import cvzone
import numpy as np

from hand_tracking import handDetector


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera.")

    detector = handDetector(detectionConf=0.8)
    pixel_spans = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    centimeters = list(range(20, 101, 5))
    a, b, c = np.polyfit(pixel_spans, centimeters, 2)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img = detector.findHands(img, draw=False)
            distance = detector.findDistance(5, 17, img)
            if distance >= 0:
                distance_cm = a * distance**2 + b * distance + c
                cvzone.putTextRect(img, f"{int(distance_cm)} cm", (20, 30), scale=2, thickness=2)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
