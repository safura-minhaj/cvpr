import cv2
import numpy as np

img = cv2.imread('input.jpg')
z = img.reshape((-1,3)).astype(np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(z, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()].reshape((img.shape))

cv2.imshow('K-Means Segmentation', res)
cv2.waitKey(0)
cv2.destroyAllWindows()