import cv2
import time

from hand_tracking import handDetector


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera.")

    detector = handDetector()
    previous_time = 0
    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img = detector.findHands(img)
            landmarks = detector.findPosition(img, draw=False)
            if landmarks:
                print(landmarks[8])

            current_time = time.time()
            fps = 1 / (current_time - previous_time) if previous_time else 0
            previous_time = current_time
            cv2.putText(img, f"fps: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
