import os.path
import cv2
import numpy as np
import random

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# cascade = cv2.CascadeClassifier()
# cap = cv2.VideoCapture("http://192.168.0.100:8080/video")

resolution = (1080, 720)

def get_frame_from_phone(IP):
    return cv2.resize(cv2.VideoCapture(IP).read()[1], resolution)

def get_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cascade.detectMultiScale(gray)
    except:
        None

def draw_faces(img, faces):
    for i in faces:
        cv2.rectangle(img, [i[0], i[1]], [i[0]+i[2], i[1]+i[2]], [0, 0, 255], 2)

def random_face_selection(img, faces):
    # print(faces)
    face = random.choice(faces)
    # print(face)
    image = img[face[1] - 10:face[1] + face[3] + 10, face[0] - 10:face[0] + face[2] + 10]
    return image

def DetectFaces(img):
    faces = get_faces(img)
    if not faces is None:
        if len(faces) >= 1:
            image = random_face_selection(img.copy(), faces)
            draw_faces(img, faces)
            return img, image
    return img, None

if __name__ == "__main__":
    while True:
        img = get_frame_from_phone()
        faces = get_faces(img)
        if not faces is None:
            draw_faces(img, faces)
            if len(faces) >= 1:
                image = random_face_selection(cv2.resize(img, resolution), faces)
                print(image)
                cv2.imwrite(os.path.join(r"C:/Users/Katze/PycharmProjects/Random_child_to_blackboard/", "face.png"), image)
                # cv2.imshow(K, face_image)
        cv2.imshow("Kakaha", img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()