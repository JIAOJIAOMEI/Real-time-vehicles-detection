# @Author  : Mei Jiaojiao
# @Time    : 2022/1/18 17:00
# @Software: PyCharm
# @File    : Hand Gesture.py

import cv2
import numpy as np
import mediapipe as mp
import math

# initialize hand detector
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

# open camera
cap = cv2.VideoCapture(0)

# set camera resolution
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

while True:
    # read camera feed
    success, img = cap.read()
    if not success:
        print("Unable to read camera feed")
        break
    if img is None:
        continue
    img = cv2.flip(img, 1)

    # detect the hands
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        # check if both hands are detected
        if len(results.multi_hand_landmarks) == 2:
            # get the landmarks of the hands
            lmList1 = results.multi_hand_landmarks[0].landmark
            lmList2 = results.multi_hand_landmarks[1].landmark

            # get the landmarks for the index fingers
            h, w, c = img.shape
            indexTip1 = (int(lmList1[8].x * w), int(lmList1[8].y * h))
            indexTip2 = (int(lmList2[8].x * w), int(lmList2[8].y * h))

            # draw circles on the index fingertips
            cv2.circle(img, indexTip1, 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, indexTip2, 15, (255, 0, 0), cv2.FILLED)

            # calculate the distance between the index fingertips of the two hands
            distance = math.sqrt(
                (indexTip2[0] - indexTip1[0]) ** 2 + (indexTip2[1] - indexTip1[1]) ** 2)

            # draw a line between the index fingertips
            cv2.line(img, indexTip1, indexTip2, (255, 0, 0), 3)

            # adjust the brightness of the camera feed based on the distance
            # brightness is from 0 to 100
            # distance is from 0 to 1000
            brightness = distance / 5
            img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

            # show the brightness on the screen
            cv2.putText(img, f"Brightness: {brightness:.2f}", (10, 40), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

            # draw the distance on the screen
            cv2.putText(img, f"Distance: {distance:.2f} pixels", (10, 70), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
        elif len(results.multi_hand_landmarks) == 1:
            # if only one hand is detected, show the message on the screen
            cv2.putText(img, "Please detect two hands", (10, 40), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)
            # show the blue circle on the index fingertip
            lmList1 = results.multi_hand_landmarks[0].landmark
            h, w, c = img.shape
            indexTip1 = (int(lmList1[8].x * w), int(lmList1[8].y * h))
            cv2.circle(img, indexTip1, 15, (255, 0, 0), cv2.FILLED)
        else:
            brightness = 0
            img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

    # show the camera feed
    cv2.imshow("Image", img)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()