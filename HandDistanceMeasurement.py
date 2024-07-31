import cv2
import HandTrackingModule as htm

####################################
wCam, hCam = 640, 480
####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


detector = htm.handDetector(detectionConf=0.8)

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    distance = detector.findDistance(5, 17, img)
    if distance != -1:
        print(distance)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
