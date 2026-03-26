import cv2

img = cv2.imread('image.jpg')

# BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# BGR to CMY (Manual)
cmy = 255 - img

cv2.imshow('HSV', hsv)
cv2.imshow('CMY', cmy)
cv2.waitKey(0)
cv2.destroyAllWindows()