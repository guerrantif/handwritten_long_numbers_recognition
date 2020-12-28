import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import random as rng
rng.seed(12345)

# reading image
# img = Image.open("img-2020128-213510.png")
# img = Image.open("img-2020129-0574.png")
# img = Image.open("img-2020129-1345.png")
img = Image.open("img-2020129-12029.png")
# img = Image.open("img-2020129-13428.png")
# img = Image.open("img-2020129-13744.png")
new_image = img

img_cv = np.array(img)
img_cv2 = np.array(img)


# new_image = ImageEnhance.Brightness(new_image).enhance(1.7)
# new_image = ImageEnhance.Contrast(new_image).enhance(0.3)

new_image = np.array(new_image)
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
new_image = Image.fromarray(new_image)

new_image = new_image.filter(ImageFilter.GaussianBlur(0.7))
# new_image = ImageEnhance.Color(new_image).enhance(0.)
new_image = ImageEnhance.Contrast(new_image).enhance(2.5)


#convert pil.image to opencv (numpy.ndarray)
#need numpy library for this
new_image = np.array(new_image)
# vertical_profile = np.sum(new_image,axis=1)
# horizontal_profile = np.sum(new_image,axis=0)

# plt.plot(vertical_profile)
# plt.show()


# cv2.imshow("Edge Detection and segmentation", new_image)
# # cv2.imshow("Edge Detection and segmentation", img_cv)
# cv2.waitKey(0)   # waits until a key is pressed
# cv2.destroyAllWindows()  # 




# new_image = np.array([new_image > 200])
# filtering with a Gaussian kernel to remove noise
# new_image = cv2.GaussianBlur(new_image, ksize=(7, 7), sigmaX=cv2.BORDER_DEFAULT)
# blurred = cv2.blur(gray, ksize=(9, 9))


# new_image = cv2.adaptiveThreshold(new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21, 2)
# ret, new_image = cv2.threshold(new_image, 220, 255, 0)

# new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
# new_image = Image.fromarray(new_image)

# new_image = ImageEnhance.Color(new_image).enhance(-0.3)


# new_image = np.array(new_image)

# # dilation
kernel = np.ones((6,6),np.uint8)
new_image = cv2.dilate(~new_image, kernel, iterations=3)
ret, new_image = cv2.threshold(new_image, 210, 255, 0)
# new_image = cv2.adaptiveThreshold(new_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3, 2)





contours, _ = cv2.findContours(new_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
boundings = list()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if x == 0 and y == 0:   # skip global contour
        continue
    boundings.append((x, y, w, h))
    # if 
    # cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 240, 0), 2)
# print(boundings)
# yigig
new_boundings = boundings[:]
for i in range(len(boundings)):
    for j in range(len(boundings)):
        if boundings[i] is boundings[j]: continue
        j_x, j_y, j_w, j_h = boundings[j]
        i_x, i_y, i_w, i_h = boundings[i]
        if j_x > i_x and j_y > i_y:
            if j_x + j_w < i_x + i_w and j_y + j_h < i_y + i_h:
                new_boundings[j] = None

new_boundings = [b for b in new_boundings if b != None]
# print(boundings)

for b in new_boundings:
    x, y, w, h = b
    cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 240, 0), 2)

for b in boundings:
    x, y, w, h = b
    cv2.rectangle(img_cv2, (x, y), (x+w, y+h), (0, 240, 0), 2)

# display input and output image
cv2.imshow("preprocessed image", new_image)
cv2.imshow("Edge Detection and segmentation", img_cv)
cv2.imshow("Edge Detection and segmentation1", img_cv2)
cv2.waitKey(0)   # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
