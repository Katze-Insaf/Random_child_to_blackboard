# import pygame
import random
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

phrases = ["че сидим?", "хахаха, бот", "не повезло, не повезло", "лашара",
           "каков второй закон Ньютона?", "ЪУЪ", "Че делать будем?",
           "Станцуй лисгинку", "Сделай сигму", "Лучше бы не пришел в школу...",
           "Скока до звонка осталось?", "А че какой следующий урок"]
# face = cv2.imread("face.png")

def GettingBrawler(face, text, video, size_of_video, timing, size, offset):
    video = cv2.VideoCapture(video)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    t = time.time()
    show = False
    # _, frame = video.read()
    i = 0
    while True:
        i += 1
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        if i < frames_count:
            _, frame = video.read()
            frame = cv2.resize(frame, size_of_video)
            if show:
                # print('OK')
                add_transparent_image(frame, cv2.resize(cv2.cvtColor(face, cv2.COLOR_RGB2RGBA), (size, size)),
                                      offset[0], offset[1])
                cv2.rectangle(frame, (offset[0] + size, offset[0] + int(size / 5)),
                              (offset[0] + size * 2, offset[0] + int(size / 5 * 4)),
                              (255, 0, 0), -size)
                cv2.putText(frame, text, (offset[0] + size + 30, offset[0] + int(size / 3 * 1.5)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Kakaxa", frame)
            # print(time.time() - t)
            if time.time() - t >= timing:
                show = True
            if cv2.waitKey(1) == 27:
                break
            time.sleep(1/fps)
            # elif cv2.waitKey(1) == ord('t'):
            #     time.sleep(2)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, frames_count - 1)
            _, frame = video.read()
            frame = cv2.resize(frame, size_of_video)
            add_transparent_image(frame, cv2.resize(cv2.cvtColor(face, cv2.COLOR_RGB2RGBA), (size, size)),
                                  offset[0], offset[1])
            cv2.rectangle(frame, (offset[0] + size, offset[0] + int(size / 5)),
                          (offset[0] + size * 2, offset[0] + int(size / 5 * 4)),
                          (255, 0, 0), -size)
            cv2.putText(frame, text, (offset[0] + size + 30, offset[0] + int(size / 3 * 1.5)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Kakaxa", frame)
            if cv2.waitKey(1) == 27:
                break



cap = cv2.VideoCapture(0)

while True:
    # image, face_image = Camera.DetectFaces("http://192.168.0.100:8080/video")
    _, image = cap.read()
    # add_transparent_image(image, start_button, 340, 560)
    cv2.imshow("Kakaxa", image)
    if cv2.waitKey(1) == 27:
        break
    elif cv2.waitKey(1) == ord("e"):
        image, face_image = Camera.DetectFaces(image)
        GettingBrawler(face_image, random.choices(phrases)[0],"C:/Users/minex/IdeaProjects/Random_child_to_blackboard/videos/ElementaryOddFlickertailsquirrel-size_restricted.gif",
                       (1080, 720), 3.25, 480, (100, 100))
cv2.destroyAllWindows()