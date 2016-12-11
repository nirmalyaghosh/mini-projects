# -*- coding: utf-8 -*-
"""
Extracts primary subject's face from frames of a specified video file.

@author: Nirmalya Ghosh
"""

import cv2
import imageio
import logging


def detect_faces(image, face_cascade_classifier):
    faces = face_cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces


def extract_frames_from_video(video_file_path,
                              extract_filename_prefix,
                              extract_image_size=(500, 500)):
    print("Reading {}".format(video_file_path))
    reader = imageio.get_reader(video_file_path, "ffmpeg")
    prefix = extract_filename_prefix
    timestamp = 0
    crop_image = True
    target_w, target_h = extract_image_size[0], extract_image_size[1]
    try:
        for num, image in enumerate(reader.iter_data()):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            file_path = "{}_{}.png".format(prefix, num)
            x, y, w, h = get_face_bounding_box(gray)
            if x > 0:
                frame_width, frame_height = image.shape[1], image.shape[0]

                if (x < frame_width / 2 and x > (target_w / 2)):
                    x, w = x - (target_w / 4), target_w
                if (y > frame_height / 2 and y > (target_h / 2)):
                    y, h = y - (target_h / 4), target_h

                if crop_image == False:
                    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    save_as_png(cropped, file_path)
                else:
                    cropped = gray[y:y + target_h, x:x + target_w]
                    save_as_png(cropped, file_path)

            if num % int(reader._meta["fps"]):
                continue
            else:
                timestamp = float(num) / reader.get_meta_data()["fps"]
    except RuntimeError,e:
        print("Reached the end or something went wrong")
        logging.exception(e)

    reader.close()


def get_face_bounding_box(image, width=500, height=500):
    # Detects the face(s) in the specified frame
    # and selects the one with largest area
    faces = detect_faces(image)
    num_faces = len(faces)
    if num_faces == 0:
        return (-1, -1, -1, -1)

    bounded_areas = [0] * num_faces
    i = 0
    for (x, y, w, h) in faces:
        bounded_areas[i] = w * h
        i += 1

    index = bounded_areas.index(max(bounded_areas))
    (x, y, w, h) = faces[index]
    return (x, y, w, h)


def init_face_cascade_classifier(haarcascade_xml_file_path):
    return cv2.CascadeClassifier(haarcascade_xml_file_path)


def save_as_png(image_array, file_path):
    # Credit : http://stackoverflow.com/a/27115931
    cv2.imwrite(file_path, image_array)
