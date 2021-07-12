import cv2
import numpy as np
import math
import mediapipe as mp


class FaceLandMarks:
    def __init__(self, img: str, image=None):
        self.img = img
        self.image = None  # read img and save it in this var
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2)

    def face_mesh(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.7) -> dict:
        # Run MediaPipe Face Mesh.
        with self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                min_detection_confidence=min_detection_confidence) as face_mesh:
            # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
            results = face_mesh.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            # Draw face landmarks of each face.
            if not results.multi_face_landmarks:
                return {}
            annotated_image = self.image.copy()
            img = mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results.multi_face_landmarks,
                connections=self.mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec)
            return {
                "origin_img": annotated_image,
                "output_img": img,
                "landmarks": results.multi_face_landmarks
            }


# dev
if __name__ == '__main__':
    print('dev')
