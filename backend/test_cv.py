import cv2
import numpy as np
from cv_engine import detect_obstacle_distance

img = np.zeros((400,400,3), dtype=np.uint8)

cv2.rectangle(img,(150,150),(250,250),(0,0,255),-1)

dist = detect_obstacle_distance(img)

print("Detected distance:", dist)