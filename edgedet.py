import cv2
import numpy as np

img = cv2.imread('input.jpg', 0)

# 1D Operator
edge_1d = cv2.filter2D(img, -1, np.array([[-1, 1]]))

# Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobel = cv2.convertScaleAbs(sobelx + sobely)

# Prewitt
px = cv2.filter2D(img, -1, np.array([[1,1,1],[0,0,0],[-1,-1,-1]]))
py = cv2.filter2D(img, -1, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
prewitt = px + py

# Roberts
rx = cv2.filter2D(img, -1, np.array([[1, 0], [0, -1]]))
ry = cv2.filter2D(img, -1, np.array([[0, 1], [-1, 0]]))
roberts = rx + ry


# Laplacian
laplacian = cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F))

cv2.imshow('Sobel', sobel)
cv2.imshow('Prewitt', prewitt)
cv2.imshow('Roberts', roberts)
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()