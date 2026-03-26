import cv2
import numpy as np

# 1. Load the image
i = cv2.imread("img1.jpg")

# 2. Convert to Grayscale (code 6) and then to float32 for math processing
g = np.float32(cv2.cvtColor(i, 6))

# 3. Compute PCA
# m = mean, e = eigenvectors. We are keeping 50 principal components.
m, e = cv2.PCACompute(g, None, 50)

# 4. Project and Back-Project
# This compresses the image data and then reconstructs it from the 50 components
p = cv2.PCAProject(g, m, e)
r = cv2.PCABackProject(p, m, e)

# 5. Display Results
# Divide by 255 because imshow expects floats to be in the range [0.0, 1.0]
cv2.imshow("Original (O)", g / 255)
cv2.imshow("PCA Reconstructed (P)", r / 255)

# 6. Cleanup
cv2.waitKey(0)
cv2.destroyAllWindows()