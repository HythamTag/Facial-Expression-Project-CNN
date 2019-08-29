import cv2
import numpy as np


def Load_Face_Detection_Cascade_Model(path):
    model = cv2.CascadeClassifier(path)
    return model


def Detect_Faces(model, image):
    return model.detectMultiScale(image, 1.3, 5)  # Default Values


def Draw_Bounding_BOX(Face_points, image, color_box):
    x_point, y_point, width, height = Face_points
    #print(Face_points)
    cv2.rectangle(image, (x_point, y_point), (x_point + width, y_point + height), color_box)


Face_Detection_XML_Path = "Models/Face_Detection_Model/haarcascade_frontalface_default.xml"
Face_Detection_Cascade_Model = Load_Face_Detection_Cascade_Model(Face_Detection_XML_Path)

cv2.namedWindow('Live')
video_capture = cv2.VideoCapture(0)

while True:
    bgr_frame = video_capture.read()[1]

    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    faces_points = Detect_Faces(Face_Detection_Cascade_Model, gray_frame) # array of array of numpy

    for face_points in faces_points:
        Draw_Bounding_BOX(face_points, bgr_frame, (255, 0, 0))

    cv2.imshow('window_frame', bgr_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
