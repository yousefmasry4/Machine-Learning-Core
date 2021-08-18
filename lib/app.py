import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
from AI.Face.face import FaceLandMarks
from AI.Face.facePoints import facePoints
from AI.Hand.Hand import HandLandMarks
import base64
from PIL import Image
from io import StringIO

app = Flask(__name__)


def readb64(base64_string):
    decoded_data = base64.b64decode(base64_string)
    np_data = np.fromstring(decoded_data, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)


@app.route('/maskImage', methods=['POST'])
def mask_image():
    print("-----------------")
    request_data = request.get_json()
    print(request_data)
    face_oval = request_data['face_oval'] if 'face_oval' in request_data else None
    left_eye = request_data['left_eye'] if 'left_eye' in request_data else None
    right_eye = request_data['right_eye'] if 'right_eye' in request_data else None
    left_eye_brow = request_data['left_eye_brow'] if 'left_eye_brow' in request_data else None
    right_eye_brow = request_data['right_eye_brow'] if 'right_eye_brow' in request_data else None
    # print(request.files , file=sys.stderr)
    print(face_oval)
    img = readb64(str(request_data['image']))
    ans = facePoints(FaceLandMarks(image=img).face_mesh())
    jans = {}
    if face_oval is not None:
        jans["face_oval"] = ans.face_oval().tolist()
    if left_eye is not None:
        jans["left_eye"] = ans.left_eye().tolist()
    if right_eye is not None:
        jans["right_eye"] = ans.right_eye().tolist()
    if left_eye_brow is not None:
        jans["left_eye_brow"] = ans.left_eye_brow().tolist()
    if right_eye_brow is not None:
        jans["right_eye_brow"] = ans.right_eye_brow().tolist()

    return jsonify({'Points': jans})


@app.route('/wrist', methods=['POST'])
def wrist():
    print("-----------------")
    request_data = request.get_json()
    img = readb64(str(request_data['image']))
    print(readb64(str(request_data['image'])))
    ans = HandLandMarks(image=img).hand_landmarks()
    return jsonify({'Points': ans})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
