import cv2
import mediapipe as mp
from numpy import ones, vstack
from numpy.linalg import lstsq


class HandLandMarks:
    def __init__(self, img=None, image=None):
        self.img = img
        if self.img is None and image is None:
            raise Exception("You should provide image")

        self.mp_hands =  mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=2)
    @staticmethod
    def midpoint(p1, p2):
      return (p1.x + p2.x) / 2, (p1.y + p2.y) / 2

    @staticmethod
    def load_image(path):
        return cv2.imread(path)
    def hand_landmarks (self)-> []:

        with self.mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
        ) as hands:
         print(self.img)
         image = cv2.imread(self.img)

         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
         image.flags.writeable = False
         results = hands.process(image)


         image.flags.writeable = True
         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if( results.multi_hand_landmarks is not None):
          wrist = results.multi_hand_landmarks[0].landmark[0]
          m1 = results.multi_hand_landmarks[0].landmark[5]
          m2 = results.multi_hand_landmarks[0].landmark[17]
          mid = self.midpoint(m1, m2)

          print(wrist)
          points = [(mid[0], mid[1]), (wrist.x, wrist.y)]
          x_coords, y_coords = zip(*points)
          A = vstack([x_coords, ones(len(x_coords))]).T
          m, c = lstsq(A, y_coords)[0]
          print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
          newX = 0
          if wrist.x > mid[0]:
             newX = wrist.x +(70/image.shape[1])
          else:
             newX = wrist.x -(70/image.shape[1])
          newY=m*newX+c

          image = cv2.circle(
                image,
                center=(int(newX* image.shape[1]), int(newY * image.shape[0])),
                radius=7,
                color=(0, 255, 0),
                thickness=-1,
            )
          cv2.imshow('MediaPipe Hands', image)

          cv2.waitKey(0)
if __name__ == '__main__':
    hm=HandLandMarks('/Users/nouromran/Documents/Augmania/Machine-Learning-Core/test/236261405_569813930722050_2309333142389947229_n.jpg')
    hm.hand_landmarks()