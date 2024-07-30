import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionConf, min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils  # TO DRAW LINES BETWEEN THE LANDMARKS

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # AS MEDIAPIPE WORKS ON RGB
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # mpDraw.draw_landmarks(img, hand_landmarks)  # FOR DRAWING POINTS
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)  # FOR DRAWING LINES

        return img


    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)  # lm will contain the coordinates
                h, w, c = img.shape # GETTING HEIGHT, WIDTH AND CHANNEL OF OUR CAMERA IMG
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)

        return lmList

    def findDistance(self, index1, index2, img):
        lmList = self.findPosition(img, draw=False)
        x1, y1, x2, y2 = lmList[index1][1], lmList[index1][2], lmList[index2][1], lmList[index2][2]
        return math.hypot((x1, y1), (x2, y2))




def main():
    cap = cv2.VideoCapture(0)
    pTime = 0 #PREVIOUS TIME
    cTime = 0 #CURRENT TIME

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList):
            print(lmList[8]) # printing positions for index finger

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, 'fps: ' + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()
