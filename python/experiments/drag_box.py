import cv2
import cvzone

from hand_tracking import handDetector


class DragRect:
    def __init__(self, position_center, size=(200, 200)):
        self.position_center = position_center
        self.size = size

    def update(self, cursor):
        cx, cy = self.position_center
        width, height = self.size
        x, y = cursor
        if cx - width // 2 < x < cx + width // 2 and cy - height // 2 < y < cy + height // 2:
            self.position_center = cursor


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = handDetector(detectionConf=0.7)
    rects = [DragRect((i * 250 + 150, 150)) for i in range(5)]
    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            color = (255, 0, 255)
            img = cv2.flip(img, 1)
            img = detector.findHands(img)
            landmarks = detector.findPosition(img, draw=False)

            if landmarks and detector.findDistance(8, 12, img) < 40:
                cursor = (landmarks[8][1], landmarks[8][2])
                for rect in rects:
                    rect.update(cursor)

            for rect in rects:
                cx, cy = rect.position_center
                width, height = rect.size
                cv2.rectangle(img, (cx - width // 2, cy - height // 2),
                              (cx + width // 2, cy + height // 2), color, cv2.FILLED)
                cvzone.cornerRect(img, (cx - width // 2, cy - height // 2, width, height))

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
