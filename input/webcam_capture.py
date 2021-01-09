import cv2
import datetime

# opens a camera for video capturing
cam = cv2.VideoCapture(0)

# creates a window
cv2.namedWindow("Long number recognition")


while True:

    # grabs, decodes and returns the next video frame
    retval, frame = cam.read()

    if not retval:
        print("Unable to grab the frame!")
        break

    # show the countinously captured frame
    cv2.imshow("Long number recognition", frame)

    # waits for a pressed key
    k = cv2.waitKey(delay=1)

    # if the ESC is pressed
    if k % 256 == 27:
        print("Exiting...")
        break
    
    # if the SPACE is pressed 
    elif k % 256 == 32:
        timestamp = datetime.datetime.now()
        img_name = "img-{}{}{}-{}{}{}.png".format(
                                          timestamp.year
                                        , timestamp.month
                                        , timestamp.day
                                        , timestamp.hour
                                        , timestamp.minute
                                        , timestamp.second)
        # saves the frame to a specified path
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

# release the capture
cam.release()

# destroy the opened windows
cv2.destroyAllWindows()