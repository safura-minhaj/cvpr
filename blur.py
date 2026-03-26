import cv2

img = cv2.imread('image.jpg')

# Gaussian Blur (Smoothing)
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# Median Blur (Salt & Pepper Noise Removal)
median = cv2.medianBlur(img, 5)

cv2.imshow('Gaussian Blur', gaussian)
cv2.imshow('Median Blur', median)
cv2.waitKey(0)
cv2.destroyAllWindows()