import cv2
import math
import mediapipe as mp
import numpy as np
from numpy.linalg import lstsq
from sympy import symbols, Eq, solve

from mediapipe.framework.formats.detection_pb2 import Detection

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def point(_3enYmen, _3enShemal, nose, lips, imgX, imgY):
    """
  1. Scale_H=msafa ma ben 3enshmal w 3en ymen
  2. Scale_V=msafa ma ben nose w lips
  3. line ma ben lips w nose
  4. get point in line by scale
  5. get rotation by m of line
  6. displacment nose and lips point by 3 points eyes and nose
  """
    # first of all implement func to get distance between two points
    getDistance = lambda x1, x2, y1, y2: ((((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** 0.5)
    Scale_H = getDistance(_3enYmen.x * imgY, _3enShemal.x * imgY, _3enYmen.y * imgX, _3enShemal.y * imgX)
    Scale_V = getDistance(nose.x * imgY, lips.x * imgY, nose.y * imgX, lips.y * imgX)
    # Calculate the coefficients. This line answers the initial question.
    # nose.x=lips.x
    # def slope_intercept(x1, y1, x2, y2):
    #     a = (y2 - y1) / (x2 - x1)
    #     b = y1 - a * x1
    #     return a, b
    # a,b=slope_intercept(nose.x * imgY,nose.y * imgX,lips.x * imgY,lips.y * imgX)
    #where y=xa+b
    #get init pos of new point
    newY=int((lips.y * imgX)+(Scale_V*3))
    newX=int(lips.x* imgY)
    #
    # angle_in_radians = math.atan(a)
    # angle_in_degrees = math.degrees(angle_in_radians)
    # print(angle_in_degrees)

    return newX,newY
    pass


# For static images:
IMAGE_FILES = []
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.2) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            continue
        annotated_image = image.copy()
        for detection in results.detections:
            print('Nose tip:')
            print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            mp_drawing.draw_detection(annotated_image, detection)
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
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
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                # print(detection.location_data.relative_keypoints[0])
                # exit(0)
                # Center coordinates
                # print(int(detection.location_data.relative_keypoints[0].x*image.shape[0]),int(detection.location_data.relative_keypoints[0].y*image.shape[0]))
                _3en_ymen=detection.location_data.relative_keypoints[1]
                _3en_shmal=detection.location_data.relative_keypoints[0]
                nose=detection.location_data.relative_keypoints[2]
                lips=detection.location_data.relative_keypoints[3]
                imgx=image.shape[0]
                imgy=image.shape[1]
                newpoint=point(_3en_ymen,_3en_shmal,nose,lips,imgx,imgy)
                center_coordinates = (newpoint[0],newpoint[1])

                # Radius of circle
                radius = 20

                # Blue color in BGR
                color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 2
                cv2.circle(image, center_coordinates, radius, color, thickness)
                # cv2.line(image, (int(nose.x * image.shape[1]), int(nose.y * image.shape[0])), (int(newpoint[0]), int(newpoint[1])),(0, 255, 0), thickness=2)
                # mp_drawing.draw_detection(image, detection)
        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
