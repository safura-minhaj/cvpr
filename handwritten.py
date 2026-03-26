import cv2
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

X, y = fetch_openml('mnist_784', return_X_y=True)

model = KNeighborsClassifier(3).fit(X/255.0, y.astype(int))

while True:
    p = input("Enter path: ")
    if p == "exit": break

    img = cv2.imread(p, 0)
    img = cv2.resize(img, (28,28))
    _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)

    pred = model.predict(img.reshape(1,784)/255.0)

    print("Digit:", pred[0])