import cv2
import math
import numpy as np

from hand_tracking import handDetector


def main():
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except ImportError as error:
        raise RuntimeError(
            "The volume-control experiment requires Windows and the optional "
            "dependencies in requirements-windows.txt."
        ) from error

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    min_volume, max_volume, _ = volume.GetVolumeRange()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = handDetector(detectionConf=0.7)
    current_volume = min_volume
    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            img = detector.findHands(img)
            landmarks = detector.findPosition(img, draw=False)
            if landmarks:
                x1, y1 = landmarks[4][1], landmarks[4][2]
                x2, y2 = landmarks[8][1], landmarks[8][2]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, center, 12, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                length = math.hypot(x2 - x1, y2 - y1)
                current_volume = float(np.interp(length, [50, 300], [min_volume, max_volume]))
                volume.SetMasterVolumeLevel(current_volume, None)

            volume_height = np.interp(current_volume, [min_volume, max_volume], [250, 0])
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, 150 + int(volume_height)), (85, 400), (0, 255, 0), cv2.FILLED)
            percentage = 100 - int(volume_height / 250 * 100)
            cv2.putText(img, f"{percentage}%", (70, 430), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 255), 2)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
