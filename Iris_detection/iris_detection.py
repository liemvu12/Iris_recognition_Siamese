import os
import cv2

# Load the cascades
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the image
img = cv2.imread('CASIA-Iris-Lamp/001/L/S2001L01.jpg')  # Replace with your image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detect eyes
eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

for (ex, ey, ew, eh) in eyes:
    # Crop the eye from the image
    eye_img = img[ey:ey+eh, ex:ex+ew]
    # Resize the cropped image to 320x280
    resized_eye = cv2.resize(eye_img, (384, 384))
    # Display the resized eye image
    cv2.imwrite('S2001L01_detec.jpg',resized_eye)
    cv2.imshow('Eye', resized_eye)
    cv2.waitKey(0)
# Test the function

cv2.destroyAllWindows()
