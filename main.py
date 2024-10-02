# import pygame
import time

import Camera
import sys
import cv2
import numpy as np

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape
    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite
    # return composite

start_button = cv2.resize(cv2.cvtColor(cv2.imread("Start.png"), cv2.COLOR_RGB2RGBA), (400, 128))
face = cv2.imread("face.png")

def GettingBrawler(face, text, video, timing, size, offset):
    video = cv2.VideoCapture(video)
    t = time.time()
    show = False
    while True:
        _, frame = video.read()
        if _:
            if show:
                print('OK')
                add_transparent_image(frame, cv2.resize(cv2.cvtColor(face, cv2.COLOR_RGB2RGBA), (size, size)),
                                      offset[0], offset[1])
            cv2.imshow("Kakaxa", frame)
            # print(time.time() - t)
            if time.time() - t >= timing:
                show = True

            if cv2.waitKey(1) == 27:
                break
            elif cv2.waitKey(1) == ord('t'):
                time.sleep(2)



while True:
    image, face_image = Camera.DetectFaces("http://192.168.0.102:8080/video")
    # add_transparent_image(image, start_button, 340, 560)
    cv2.imshow("Kakaxa", image)
    if cv2.waitKey(1) == 27:
        break
    elif cv2.waitKey(1) == ord("a"):
        GettingBrawler(face_image, "хахаха лох","МЕНЯ ЗАДОЛБАЛА ЭТА ПЕСНЯ #Снежа.mp4", 1, 256, (250, 540))
cv2.destroyAllWindows()