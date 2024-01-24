import cv2
import numpy as np
import tkinter as tk
from pynput.mouse import Button, Controller

mouse = Controller()

root = tk.Tk()
screenx = root.winfo_screenwidth()
screeny = root.winfo_screenheight()
capturex, capturey = 700, 500  # captures this size frame

cap = cv2.VideoCapture(0)
cap.set(3, capturex)
cap.set(4, capturey)

# Adjusted HSV color range
lb = np.array([15, 80, 80])
ub = np.array([30, 255, 255])

# Adjusted contour area thresholds
open_hand_lower_limit = 5000
open_hand_upper_limit = 20000
closed_hand_lower_limit = 1000
closed_hand_upper_limit = 8000

kernelOpen = np.ones((3, 3), np.uint8)  # Adjusted kernel size for opening
kernelClose = np.ones((15, 15), np.uint8)  # Adjusted kernel size for closing

cd = 0

while True:
    ret, frame = cap.read()

    # Use HSV of yellow to detect only yellow color
    imgSeg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masking and filtering all shades of yellow
    mask = cv2.inRange(imgSeg, lb, ub)

    # Apply morphology to avoid noise
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    final = maskClose
    conts, _ = cv2.findContours(maskClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(conts) != 0:
        b = max(conts, key=cv2.contourArea)
        west = tuple(b[b[:, :, 0].argmin()][0])
        east = tuple(b[b[:, :, 0].argmax()][0])
        north = tuple(b[b[:, :, 1].argmin()][0])
        south = tuple(b[b[:, :, 1].argmax()][0])
        centre_x = (west[0] + east[0]) / 2
        centre_y = (north[0] + south[0]) / 2

        cv2.drawContours(frame, [b], -1, (0, 255, 0), 3)
        cv2.circle(frame, west, 6, (0, 0, 255), -1)
        cv2.circle(frame, east, 6, (0, 0, 255), -1)
        cv2.circle(frame, north, 6, (0, 0, 255), -1)
        cv2.circle(frame, south, 6, (0, 0, 255), -1)
        cv2.circle(frame, (int(centre_x), int(centre_y)), 6, (255, 0, 0), -1)

        bint = int(cv2.contourArea(b))

        if open_hand_lower_limit <= bint <= open_hand_upper_limit:  # Hand is open
            mouse.release(Button.left)
            cv2.circle(frame, (int(centre_x), int(centre_y)), 6, (255, 0, 0), -1)
            mouse.position = (screenx - (centre_x * screenx / capturex),
                              screeny - (centre_y * screeny / capturey))

        elif closed_hand_lower_limit <= bint <= closed_hand_upper_limit:  # Hand is closed
            cv2.circle(frame, (int(centre_x), int(centre_y)), 10, (255, 255, 255), -1)
            mouse.position = (screenx - (centre_x * screenx / capturex),
                              screeny - (centre_y * screeny / capturey))
            mouse.press(Button.left)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):  # Exiting
        break

cap.release()
cv2.destroyAllWindows()
