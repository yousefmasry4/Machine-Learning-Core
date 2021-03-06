import cv2
import mediapipe as mp
from numpy import ones, vstack
from numpy.linalg import lstsq
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:


        mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS)

        cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
def midpoint(p1, p2):
    return (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.7,
    max_num_hands=1,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,)
        wrist = results.multi_hand_landmarks[0].landmark[0]
        m1 = results.multi_hand_landmarks[0].landmark[5]
        m2 = results.multi_hand_landmarks[0].landmark[17]
        mid = midpoint(m1, m2)

        points = [(mid[0], mid[1]), (wrist.x, wrist.y)]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords, ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        newX = 0
        def getLen():
            return (math.dist([wrist.x, wrist.y], [mid[0], mid[1]])* image.shape[1])/2
        if wrist.x > mid[0]:
            newX = wrist.x + (getLen() / image.shape[1])
        else:
            newX = wrist.x - (getLen() / image.shape[1])
        newY = m * newX + c
        print(getLen() )

        image = cv2.circle(
            image,
            center=(int(newX * image.shape[1]), int(newY * image.shape[0])),
            radius=7,
            color=(0, 255, 0),
            thickness=-1,
        )
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
