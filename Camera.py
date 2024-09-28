import cv2
import numpy as np
import random

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture("http://192.168.0.100:8080/video")

resolution = (480, 320)

def get_frame_from_phone():
    return cv2.resize(cap.read()[1], resolution)

def get_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cascade.detectMultiScale(gray)
    except:
        None

def draw_faces(img, faces):
    for i in faces:
        cv2.rectangle(img, [i[0], i[1]], [i[0]+i[2], i[1]+i[2]], [0, 0, 255], 2)

def random_face_selection(faces):
    # print(faces)
    face = random.choice(faces)
    # print(face)
    image = img[face[0]:face[0]+face[2], face[1]:face[1]:face[3]]
    return image

if __name__ == "__main__":
    while True:
        img = get_frame_from_phone()
        faces = get_faces(img)
        if not faces is None:
            draw_faces(img, faces)
            if len(faces) >= 1:
                img = random_face_selection(faces)
                # cv2.imshow(K, face_image)
        cv2.imshow("Kakaha", img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()