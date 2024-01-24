import cv2
import numpy as np
import wx
from pynput.mouse import Button, Controller

mouse = Controller()

app = wx.App(False)
(screenx, screeny) = wx.GetDisplaySize()
(capturex, capturey) = (400, 300)  # captures this size frame

cap = cv2.VideoCapture(0)
cap.set(3, capturex)
cap.set(4, capturey)

kernelOpen = np.ones((5, 5), np.uint8)  # if noise are present other than yellow area
kernelClose = np.ones((20, 20), np.uint8)  # if noise are present in yellow area

# HSV color range which should be detected
lb = np.array([20, 100, 100])
ub = np.array([120, 255, 255])

cd = 0

while True:
    ret, frame = cap.read()

    # use HSV of yellow to detect only yellow color
    imgSeg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # masking and filtering all shades of yellow
    mask = cv2.inRange(imgSeg, lb, ub)

    # apply morphology to avoid noise
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    final = maskClose
    _, conts, _ = cv2.findContours(maskClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(conts) != 0:  # draws the contours of that object which has the highest
        b = max(conts, key=cv2.contourArea)
        west = tuple(b[b[:, :, 0].argmin()][0])  # gives the co-ordinate of the left extreme of contour
        east = tuple(b[b[:, :, 0].argmax()][0])  # above for east i.e right
        north = tuple(b[b[:, :, 1].argmin()][0])
        south = tuple(b[b[:, :, 1].argmax()][0])
        centre_x = (west[0] + east[0]) / 2
        centre_y = (north[0] + south[0]) / 2

        cv2.drawContours(frame, [b], -1, (0, 255, 0), 3)
        cv2.circle(frame, west, 6, (0, 0, 255), -1)
        cv2.circle(frame, east, 6, (0, 0, 255), -1)
        cv2.circle(frame, north, 6, (0, 0, 255), -1)
        cv2.circle(frame, south, 6, (0, 0, 255), -1)
        cv2.circle(frame, (int(centre_x), int(centre_y)), 6, (255, 0, 0), -1)  # plots centre of the area

        bint = int(cv2.contourArea(b))

        if 8000 <= bint <= 18000:  # hand is open
            mouse.release(Button.left)
            cv2.circle(frame, (int(centre_x), int(centre_y)), 6, (255, 0, 0), -1)  # plots centre of the area
            mouse.position = (screenx - (centre_x * screenx / capturex),
                              screeny - (centre_y * screeny / capturey))

        elif 2000 <= bint <= 7000:  # hand is closed
            cv2.circle(frame, (int(centre_x), int(centre_y)), 10, (255, 255, 255), -1)  # plots centre of the area
            mouse.position = (screenx - (centre_x * screenx / capturex),
                              screeny - (centre_y * screeny / capturey))
            mouse.press(Button.left)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):  # exiting
        break

cap.release()
cv2.destroyAllWindows()
