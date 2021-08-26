import cv2
import numpy as np
import math
import mediapipe as mp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
mp_hands = mp.solutions.hands
ai = mp.solutions.hands

class HandLandMarks:
    def __init__(self, img=None, image=None):
        self.img = img
        if self.img is None and image is None:
            raise Exception("You should provide image")
        if image is not None:
            self.image = image
        else:
            self.image = HandLandMarks.load_image(self.img)

        self.mp_hands =ai
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
        #self.drawing_styles = mp.solutions.drawing_styles

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

    def hand_landmarks(self, static_image_mode=True, min_detection_confidence=0.7) -> []:

        # Run MediaPipe Hands.
        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7) as hands:

                # Convert the BGR image to RGB, flip the image around y-axis for correct
                # handedness output and process it with MediaPipe Hands.
                results = hands.process(cv2.flip(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), 1))

                # Print handedness (left v.s. right hand).
                #print(f'Handedness of {name}:')
                print(results.multi_handedness)



                # Draw hand landmarks of each hand.
                #print(f'Hand landmarks of {name}:')
                image_hight, image_width, _ = self.image.shape
                annotated_image = cv2.flip(self.image.copy(), 1)
                for hand_landmarks in results.multi_hand_landmarks:
                    # Print index finger tip coordinates.
                    return (
                        f'Index finger tip coordinate: (',
                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
                    )
                    '''self.mp_drawing.draw_landmarks(
                        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmark_style(),
                        drawing_styles.get_default_hand_connection_style())'''


if __name__ == '__main__':
    fm = HandLandMarks(img="/Users/nouromran/Documents/Augmania/Machine-Learning-Core/test/IMG_7079.png")

    #fm.show_image("model1.png")
    print(fm.hand_landmarks())
