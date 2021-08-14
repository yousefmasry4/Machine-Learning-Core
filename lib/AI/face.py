import cv2
import numpy as np
import math
import mediapipe as mp

from facePoints import facePoints
# hi nour omran

ai = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
class FaceLandMarks:
    def __init__(self, img=None, image=None):
        self.img = img
        if self.img is None and image is None:
            raise Exception("You should provide image")
        if image is not None:
            self.image = image  # read img and save it in this var
        else:
            self.image = FaceLandMarks.load_image(self.img)

        self.mp_face_mesh = ai
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=2)

    @staticmethod
    def load_image(path):
        return cv2.imread(path)

    @staticmethod
    def show_image(path=None, img=None):
        image = FaceLandMarks.load_image(path) if path is not None else img
        D_HEIGHT = 1024
        D_WIDTH = 1024
        height, width = image.shape[:2]
        if height < width:
            image = cv2.resize(image, (D_WIDTH, math.floor(height / (width / D_WIDTH))))
        else:
            image = cv2.resize(image, (math.floor(width / (height / D_HEIGHT)), D_HEIGHT))
        cv2.imshow('img', image)
        cv2.waitKey(0)

    def face_mesh(self, static_image_mode=True, min_detection_confidence=0.7) -> []:
        # Run MediaPipe Face Mesh.
        with self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=1,
                min_detection_confidence=min_detection_confidence) as face_mesh:
            # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
            results = face_mesh.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            # Draw face landmarks of each face.
            if not results.multi_face_landmarks:
                return {}
            annotated_image = self.image.copy()
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec)
            return [
                    [int(data_point.x * self.image.shape[0]), int(data_point.y * self.image.shape[1])]
                    for data_point in results.multi_face_landmarks[0].landmark
                ]
    def hand_landmarks(self, static_image_mode=True, min_detection_confidence=0.7) -> []:

        # Run MediaPipe Hands.
        with mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence) as hands:

                results = hands.process(cv2.flip(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 1))

                return [
                    [int(data_point.x * self.image.shape[1]), int(data_point.y * self.image.shape[0])]
                    for data_point in results.multi_hand_landmarks[0].landmark
                ]

# dev
if __name__ == '__main__':
    fm = FaceLandMarks(img="../../test/236581335_4599727816746406_4505042074098827672_n.jpg")
    # FaceLandMarks.show_image(img=fm.image)
    # FaceLandMarks.show_image(path="test/1.jpg")
    # ans=fm.face_points()
    # FaceLandMarks.show_image(img=ans["output_img"])
    landmarks_img =fm.image
    points= fm.hand_landmarks()
    for k, landmark in enumerate(points, 1):
        print(landmark)
        landmarks_img = cv2.circle(
            landmarks_img,
            center=(int(landmark[0]), int(landmark[1])),
            radius=3,
            color=(0, 255, 0),
            thickness=-1,
        )
        # draw landmarks' labels
        landmarks_img = cv2.putText(
            img=landmarks_img,
            text=str(k),
            org=(int(landmark[0]) + 5, int(landmark[1]) + 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
        )
    cv2.imshow('img', landmarks_img)
    cv2.waitKey(0)