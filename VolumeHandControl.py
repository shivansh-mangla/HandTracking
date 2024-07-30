import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math

###### PYCAW CODE FOR VOLUME CONTROL-
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)


volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0

####################################
wCam, hCam = 640, 480
####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionConf=0.7)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList):
        # 4-> thumb, 8-> index finger
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cx, cy = (x1+x2) // 2, (y1+y2) // 2  # CORDINATES OF CENTRE

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Hand range 50 - 300
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)

    volHeight = np.interp(vol, [minVol, maxVol], [250, 0])
    cv2.rectangle(img, (50,150), (85, 400), (0,255,0), 3)
    cv2.rectangle(img, (50,150+int(volHeight)), (85, 400), (0,255,0), cv2.FILLED)

    cv2.putText(img, f"{100 - int((volHeight/250)*100)}%", (70, 430), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 255), 2)

    cv2.imshow("IMAGE", img)
    cv2.waitKey(1)
