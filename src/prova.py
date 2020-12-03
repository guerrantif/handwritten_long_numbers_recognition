import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random as rng
rng.seed(12345)

# reading image
img = cv.imread("numbers.jpg")

# converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)


# filtering with a Gaussian kernel to remove noise
blurred = cv.GaussianBlur(gray, ksize=(17, 17), sigmaX=cv.BORDER_DEFAULT)

thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    if x == 0 and y == 0:   # skip global contour
        continue
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 240, 0), 2)

# display input and output image
cv.imshow("Edge Detection and segmentation", img)
cv.waitKey(0)   # waits until a key is pressed
cv.destroyAllWindows()  # destroys the window showing image
