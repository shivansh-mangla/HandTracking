import cv2
import cvzone

import HandTrackingModule
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cx, cy, w, h = 150, 150, 200, 200

detector = htm.handDetector(detectionConf=0.7)

class dragRect():
    def __init__(self, positionCentre, size=(200, 200)):
        self.positionCentre = positionCentre
        self.size = size

    def update(self, cursor):
        cx, cy = self.positionCentre
        w, h = self.size

        if cx-w//2 < x < cx+w//2 and cy-h//2 < y < cy+h//2:
            defaultColor = (0, 255, 0)
            self.positionCentre = cursor

rectList = []
for i in range(5):
    rectList.append(dragRect((i*250 + 150, 150)))

while True:
    success, img = cap.read()
    defaultColor = (255, 0, 255)  # When not dragging
    img = cv2.flip(img, 1)  #TO FLIP THE IMAGE (MIRROR)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # print(lmList[8]) # printing positions for index finger
        distance = detector.findDistance(8, 12, img)
        # print(distance)

        if distance < 40:
            cursor = (lmList[8][1], lmList[8][2])  # X AND Y OF CURSOR
            x, y = cursor[0], cursor[1]
            for rect in rectList:
                rect.update(cursor)


    for rect in rectList:
        cx, cy = rect.positionCentre
        w, h = rect.size
        cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), defaultColor, cv2.FILLED)
        cvzone.cornerRect(img, (cx-w//2, cy-h//2, w, h))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
