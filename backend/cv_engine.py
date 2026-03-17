import cv2
import numpy as np

def detect_obstacle_distance(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    full_mask = mask1 + mask2
    kernel = np.ones((5,5), np.uint8)
    full_mask = cv2.erode(full_mask, kernel, iterations=1)
    full_mask = cv2.dilate(full_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 1000:
            x, y, w, h = cv2.boundingRect(largest)
            distance = 500 / w 
            return round(distance, 2), (x, y, w, h)
    return 100.0, None