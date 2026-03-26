import cv2

img = cv2.imread('image.jpg')

for i in range(8):
    bit_plane = (img >> i) & 1
    cv2.imshow(f'Bit Plane {i}', bit_plane * 255)

cv2.waitKey(0)
cv2.destroyAllWindows()
