import cv2
import numpy as np
import math
import mediapipe as mp


class FaceLandMarks:
    def __init__(self, img=None, image=None):
        self.img = img
        if self.img is None and image is None:
            raise Error("You should provide image")
        if image is not None:
            self.image = image  # read img and save it in this var
        else:
            self.image = FaceLandMarks.load_image(self.img)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=2)

    @staticmethod
    def load_image(path):
        return cv2.imread(path)

    @staticmethod
    def show_image(path=None, img=None):
        image = FaceLandMarks.load_image(path) if path is not None else img
        D_HEIGHT = 480
        D_WIDTH = 480
        print(image)
        print("----------------------------------")
        height, width = image.shape[:2]
        if height < width:
            image = cv2.resize(image, (D_WIDTH, math.floor(height / (width / D_WIDTH))))
        else:
            image = cv2.resize(image, (math.floor(width / (height / D_HEIGHT)), D_HEIGHT))
        cv2.imshow('img', image)
        cv2.waitKey(0)

    def face_mesh(self, static_image_mode=True, min_detection_confidence=0.7) -> dict:
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
            return {
                "origin_img": self.image,
                "output_img": annotated_image,
                "landmarks": results.multi_face_landmarks
            }

    def face_points(self, min_detection_confidence=0.7, model_selection=0) -> dict:
        with self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence, model_selection=model_selection) as face_detection:
            resultsLands = face_detection.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            if not resultsLands.detections:
                return {}
            annotated_image = self.image.copy()
            self.mp_drawing.draw_detection(annotated_image, resultsLands.detections[0])
            return {
                "origin_img": self.image,
                "output_img": annotated_image,
                "landmarks": resultsLands.detections
            }


# dev
if __name__ == '__main__':
    fm = FaceLandMarks(img="test/1.jpg")
    # FaceLandMarks.show_image(img=fm.image)
    # FaceLandMarks.show_image(path="test/1.jpg")
    '''
    ans=fm.face_points()
    print(ans)
    FaceLandMarks.show_image(img=ans["output_img"])
    '''
    ans=fm.face_mesh()
    print(ans)
    FaceLandMarks.show_image(img=ans["output_img"])

