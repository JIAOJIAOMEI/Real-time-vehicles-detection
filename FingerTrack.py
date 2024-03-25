# @Author  : Mei Jiaojiao
# @Time    : 2024/1/18 17:00
# @Software: PyCharm
# @File    : FingerTrack.py
import cv2
from cvzone.HandTrackingModule import HandDetector

# initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# open camera
cap = cv2.VideoCapture(0)

# set camera resolution
cap.set(3, 1280) # width
cap.set(4, 720)  # height

# initialize left and right hand variables
leftHand = None
rightHand = None

while True:
    # read camera feed
    success, img = cap.read()
    if not success:
        print("Unable to read camera feed")
        break
    if img is None:
        continue
    img = cv2.flip(img, 1)

    # detect and draw the hands
    hands, img = detector.findHands(img)

    if hands:
        if len(hands) == 1:
            # if only one hand is detected, get the number of fingers up
            if hands[0]['type'] == 'Left':
                leftHand = hands[0]
                leftFingers = detector.fingersUp(leftHand)
                rightFingers = [0] * 5
            elif hands[0]['type'] == 'Right':
                rightHand = hands[0]
                rightFingers = detector.fingersUp(rightHand)
                leftFingers = [0] * 5
        elif len(hands) == 2:
            # if two hands are detected, get the number of fingers up for each hand
            leftHand = hands[0] if hands[0]['type'] == 'Left' else hands[1]
            rightHand = hands[0] if hands[0]['type'] == 'Right' else hands[1]
            leftFingers = detector.fingersUp(leftHand)
            rightFingers = detector.fingersUp(rightHand)

        fingerSum = sum(leftFingers) + sum(rightFingers)

        # draw the number of fingers up on the screen
        cv2.putText(img, f"L:{leftFingers} R:{rightFingers} Sum:{fingerSum}", (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2)

    # show the camera feed
    cv2.imshow("Image", img)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()