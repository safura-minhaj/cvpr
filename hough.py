import cv2
import numpy as np

img = cv2.imread('image.jpg')
edges = cv2.Canny(img, 50, 150)

# Probabilistic Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Hough Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()