import cv2
import mediapipe as mp
from numpy import ones, vstack
from numpy.linalg import lstsq
import math
import numpy as np
from numpy.linalg import norm
from sympy import symbols, Eq, solve

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


def limitt(n):
    if n[0] <= 1 and n[0] >= 0:
        return n[0]


def Dpoint(c, a, b, r):
    pi = 3.1415926535897932384626433832795
    degree = 2 * pi / 360
    radian = 1 / degree
    radius = r
    a0 = np.multiply(a, degree)
    b0 = np.multiply(b, degree)
    c0 = np.multiply(c, degree)
    aa = [1.0, math.cos(c0[0])]
    ba = [1.0, math.cos(c0[0])]
    ca = [1.0, math.cos(c0[0])]
    A = np.multiply(np.multiply(a0, aa), radius)
    B = np.multiply(np.multiply(b0, ba), radius)
    C = np.multiply(np.multiply(c0, ca), radius)
    V = A - C
    U = B - A
    e = (B - A).transpose()
    g = A - C
    rsquare = norm(e + g) ** 2
    alpha = np.sum((np.multiply(U, U)))
    beta = np.sum((np.multiply(U, V)))
    gamma = np.sum((np.multiply(V, V))) - rsquare
    t = []
    t.append((-beta) + math.sqrt(np.multiply(beta, beta) - np.multiply(alpha, gamma)))
    t.append((-beta) - math.sqrt(np.multiply(beta, beta) - np.multiply(alpha, gamma)))
    return np.multiply(np.outer(a0 + (b0 - a0), limitt(t)), radian)


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
                # mp_drawing.draw_landmarks(
                #     image,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS, )
                wrist = results.multi_hand_landmarks[0].landmark[0]


                p1 = results.multi_hand_landmarks[0].landmark[4]
                p2 = results.multi_hand_landmarks[0].landmark[20]

                m1 = results.multi_hand_landmarks[0].landmark[5]
                m2 = results.multi_hand_landmarks[0].landmark[17]
                mid = midpoint(m1, m2)
                print("m1 :", m1)
                print("m2 :", m2)
                print("mid :", mid)
                print("wrist :", wrist)
                print("shapeY :", image.shape[1])
                print("shapeX :", image.shape[0])

                points = [(mid[0] * image.shape[1], mid[1] * image.shape[0]),
                          (wrist.x * image.shape[1], wrist.y * image.shape[0])]
                x_coords, y_coords = zip(*points)
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords)[0]
                print("m , c = ", m, c)
                newX = 0


                def getLen():
                    xx = (math.dist([wrist.x, wrist.y], [mid[0], mid[1]]) * image.shape[1]) / 2
                    xx = 70 if xx > 70 else xx
                    return xx

                print("getLen :", getLen())
                if wrist.x < mid[0]:
                    newX = wrist.x + (getLen() / image.shape[1])
                else:
                    newX = wrist.x - (getLen() / image.shape[1])
                newY = m * newX + c

                x, y = symbols('x y')

                eq1 = Eq(x * m - y, -1 * c)
                eq2 = Eq((x - (wrist.x * image.shape[1])) ** 2 + (y - (wrist.y * image.shape[0])) ** 2, getLen() ** 2)
                ans = solve((eq1, eq2), (x, y))
                print(ans)
                # c, a, b = (wrist.x, wrist.y), (wrist.x, wrist.y), (newX, newY)
                # r = getLen()
                #
                #
                # # TODO Get intersection
                # temp = Dpoint(c, a, b, r)
                # newX = temp[0][0]
                # newY = temp[1][0]
                # print(Dpoint((48.137024, 11.575249), (48.139115, 11.578081), (48.146303, 11.593102), 1000.0))
                print((int(ans[0][0]), int(ans[0][1])))
                try:
                    image = cv2.circle(
                        image,
                        center=(int(ans[0][0]), int(ans[0][1])),
                        radius=10,
                        color=
                        (0, 255, 0) if math.dist([0,0],[p1.x,p1.y]) < math.dist([0,0],[p2.x,p2.y]) else (1000, 55, 200),
                        thickness=-1,
                    )
                except:
                    print("i")

                # cv2.line(image,(int(mid[0] * image.shape[1]), int(mid[1] * image.shape[0])) , (int(newX),int(newY)),
                #
                #           (0, 255, 0), thickness=2)
                # cv2.circle(image,(int(wrist.x* image.shape[1]),int(wrist.y* image.shape[0])), int(getLen()), (0,0,255), 0)
                print("---------------------------")
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
