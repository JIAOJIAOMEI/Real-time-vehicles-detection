# @Author  : Mei Jiaojiao
# @Time    : 2022/1/25 16:56
# @Software: PyCharm
# @File    : VehicleDetection.py

import cv2
print(cv2.__version__)

import numpy as np

# Open the video file for processing
cap = cv2.VideoCapture('VehicleDetection.mp4')

# Create MOG object for background subtraction
mog = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Define minimum width and height for a detected car
min_w = 90
min_h = 90

# Define the detection line position
line_high = 600

# Define the offset from the detection line for a car to be counted
offset = 7

# Initialize an empty list to store detected car positions
cars = []

# Initialize the car count to zero
carno = 0


# Function to calculate the center point of a bounding rectangle
def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = int(x) + x1
    cy = int(y) + y1
    return cx, cy


# Process each frame in the video
while True:
    ret, frame = cap.read()
    if ret == True:
        # Convert the frame to grayscale and apply Gaussian blur for noise reduction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)

        # Apply MOG for background subtraction and morphological operations for noise reduction
        mask = mog.apply(blur)
        erode = cv2.erode(mask, kernel)
        dialte = cv2.dilate(erode, kernel, iterations=2)
        close = cv2.morphologyEx(dialte, cv2.MORPH_CLOSE, kernel)

        # Find contours in the processed image
        contours, h = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the detection line on the frame
        cv2.line(frame, (10, line_high), (1200, line_high), (255, 255, 0), 3)

        # Process each detected contour
        for contour in contours:
            # Calculate the bounding rectangle of the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            # Check if the bounding rectangle meets the minimum size requirement
            is_valid = (w >= min_w) and (h >= min_h)
            if not is_valid:
                continue

            # Draw a rectangle around the detected car and add its center point to the car list
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cpoint = center(x, y, w, h)
            cars.append(cpoint)
            cv2.circle(frame, (cpoint), 5, (0, 0, 255), -1)

            # Check if the car has crossed the detection line
            for (x, y) in cars:
                if (line_high - offset) < y < (line_high + offset):
                    # Remove the car from the list and increment the car count
                    carno += 1
                    cars.remove((x, y))

        # Draw the car count on the frame and display it
        cv2.putText(frame, 'Vehicle Count:' + str(carno), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,
        0, 255), 5)
        cv2.imshow('frame', frame)

    # Wait for the user to press a key, and exit if the key is "esc"
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the resources and close all windows
cap.release()
cv2.destroyAllWindows()