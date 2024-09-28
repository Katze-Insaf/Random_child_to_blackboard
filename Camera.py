import cv2
import numpy as np

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray)
    for i in faces:
        cv2.rectangle(img, [i[0], i[1]], [i[0]+i[2], i[1]+i[2]], [0, 0, 255], 5)

    cv2.imshow("Kakaha", img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()