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
                static_image_mode=static_image_mode, min_detection_confidence=0.5, model_complexity=1) as holistic:
                # Convert the BGR image to RGB and process it with MediaPipe Pose.
                results = holistic.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))


                # Print nose coordinates.                results.
                image_hight, image_width, _ = self.image.shape
                print(results.pose_landmarks.landmark)
                # if results.pose_landmarks:
                #     return [
                #             [int(data_point.x * self.image.shape[0]), int(data_point.y * self.image.shape[1])]
                #             for data_point in results.pose_landmarks.landmark
                #         ]
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
                HolisticLandMarks.show_image(img=annotated_image)




if __name__ == '__main__':
    fm = HolisticLandMarks(img="../../../test/236581335_4599727816746406_4505042074098827672_n.jpg")
    
    landmarks_img=HolisticLandMarks.load_image(path="../../../test/236581335_4599727816746406_4505042074098827672_n.jpg")
    # print(fm.holistic_landmarks())
    # for k, landmark in enumerate(fm.holistic_landmarks(), 1):
    #     print(landmark)
    #     landmarks_img = cv2.circle(
    #         landmarks_img,
    #         center=(int(landmark[0]), int(landmark[1])),
    #         radius=3,
    #         color=(0, 255, 0),
    #         thickness=-1,
    #     )
    #     # draw landmarks' labels
    #     landmarks_img = cv2.putText(
    #         img=landmarks_img,
    #         text=str(k),
    #         org=(int(landmark[0]) + 5, int(landmark[1]) + 5),
    #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=0.5,
    #         color=(0, 255, 0),
    #     )
    print(fm.holistic_landmarks())
