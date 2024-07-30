import cv2

import HandTrackingModule
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cx, cy, w, h = 150, 150, 200, 200

detector = htm.handDetector(detectionConf=0.7)

while True:
    success, img = cap.read()
    defaultColor = (255, 0, 255)
    img = cv2.flip(img, 1)  #TO FLIP THE IMAGE (MIRROR)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        # print(lmList[8]) # printing positions for index finger
        cursor = (lmList[8][1], lmList[8][2])
        x, y = cursor[0], cursor[1]

        distance = detector.findDistance(8, 12, img)
        print(distance)

        if cx-w//2 < x < cx+w//2 and cy-h//2 < y < cy+h//2:
            defaultColor = (0, 255, 0)
            cx, cy = cursor

    cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), defaultColor, cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
