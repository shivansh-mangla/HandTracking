import cv2
import cvzone

import HandTrackingModule as htm
import numpy as np

####################################
wCam, hCam = 640, 480
####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectionConf=0.8)


""" x-> distance value between 2 points on hand,   y-> distance of hand from screen """
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coeff = np.polyfit(x, y, 2)  # Fitting these to a 2 degree polynomial
A, B, C = coeff

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)

    distance = detector.findDistance(5, 17, img)  # Finding distance between indexes 5 and 17 on hand
    if distance != -1:
        # print(distance)
        distanceCm = A*(distance**2) + B*(distance) + C
        print(distanceCm)
        cvzone.putTextRect(img, f'{int(distanceCm)} cm', (20, 30), scale=2, thickness=2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
