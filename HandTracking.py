import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils  # TO DRAW LINES BETWEEN THE LANDMARKS

pTime = 0 #PREVIOUS TIME
cTime = 0 #CURRENT TIME

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # AS MEDIAPIPE WORKS ON RGB
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                # print(id, lm)  # lm will contain the coordinates
                h, w, c = img.shape # GETTING HEIGHT, WIDTH AND CHANNEL OF OUR CAMERA IMG
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                # if id == 0:
                #     cv2.circle(img, (cx, cy), 20, (255, 0, 255), cv2.FILLED)

            # mpDraw.draw_landmarks(img, hand_landmarks)  # FOR DRAWING POINTS
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)  # FOR DRAWING LINES


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, 'fps: ' + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
