import cv2

import HandTrackingModule
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionConf=0.7)

while True:
    success, img = cap.read()
    defaultColor = (255, 0, 255)
    img = cv2.flip(img, 1)  #TO FLIP THE IMAGE (MIRROR)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList):
        # print(lmList[8]) # printing positions for index finger
        x, y = lmList[8][1], lmList[8][2]
        if 100 < x < 300 and 100 < y < 300:
            defaultColor = (0, 255, 0)

    cv2.rectangle(img, (100, 100), (300, 300), defaultColor, cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
