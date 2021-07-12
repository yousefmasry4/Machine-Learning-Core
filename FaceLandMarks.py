import cv2
import numpy as np
import math
import mediapipe as mp


class FaceLandMarks:
    def __init__(self, img):
        self.img = img
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)


# dev
if __name__ == '__main__':
    print('dev')
