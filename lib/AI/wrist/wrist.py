import cv2
import numpy as np
import math

import mediapipe as mp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
mp_holistic = mp.solutions.holistic
ai = mp.solutions.holistic
class HolisticLandMarks:
    def __init__(self, img=None, image=None):
        self.img = img
        if self.img is None and image is None:
            raise Exception("You should provide image")
        if image is not None:
            self.image = image
        else:
            self.image = HolisticLandMarks.load_image(self.img)

        self.mp_holistic =ai
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=2)

    @staticmethod
    def load_image(path):
        return cv2.imread(path)

    @staticmethod
    def show_image(path=None, img=None):
        image = HolisticLandMarks.load_image(path) if path is not None else img
        D_HEIGHT = 1024
        D_WIDTH = 1024
        height, width = image.shape[:2]
        if height < width:
            image = cv2.resize(image, (D_WIDTH, math.floor(height / (width / D_WIDTH))))
        else:
            image = cv2.resize(image, (math.floor(width / (height / D_HEIGHT)), D_HEIGHT))
        cv2.imshow('img', image)
        cv2.waitKey(0)

    def holistic_landmarks(self, static_image_mode=True, min_detection_confidence=0.7) -> []:

        with mp_holistic.Holistic(
                static_image_mode=static_image_mode, min_detection_confidence=0.5, model_complexity=2) as holistic:
                # Convert the BGR image to RGB and process it with MediaPipe Pose.
                results = holistic.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

                # Print nose coordinates.
                image_hight, image_width, _ = self.image.shape
                if results.pose_landmarks:
                    return(
                        f'Nose coordinates: ('
                        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
                        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
                    )
                # Draw pose landmarks.
                annotated_image = self.image.copy()
                self.mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=results.face_landmarks,
                    connections=mp_holistic.FACE_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=results.pose_landmarks,
                    connections=mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)



if __name__ == '__main__':
    fm = HolisticLandMarks(img="model1.png")

    #fm.show_image("model1.png")
    print(fm.holistic_landmarks())
