import cv2
import datetime

cam = cv2.VideoCapture(0)

cv2.namedWindow("videoframe")

while True:

    ret, frame = cam.read()

    if not ret:
        print("failed to grab frame")
        break

    cv2.imshow("videoframe", frame)

    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Exiting")
        break

    elif k % 256 == 32:
        # SPACE pressed
        timestamp = datetime.datetime.now()
        img_name = "img-{}{}{}-{}{}{}.png".format(
                                          timestamp.year
                                        , timestamp.month
                                        , timestamp.day
                                        , timestamp.hour
                                        , timestamp.minute
                                        , timestamp.second)
        
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

cam.release()

cv2.destroyAllWindows()