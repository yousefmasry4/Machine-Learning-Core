import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
from AI.face import FaceLandMarks
from AI.facePoints import facePoints
app = Flask(__name__)
#hi yousseff
@app.route('/maskImage', methods=['POST'])
def mask_image():
    face_oval = request.form.get('face_oval') is not None
    left_eye = request.form.get('left_eye') is not None
    right_eye = request.form.get('right_eye') is not None
    left_eye_brow = request.form.get('left_eye_brow') is not None
    right_eye_brow = request.form.get('right_eye_brow') is not None
    # print(request.files , file=sys.stderr)
    print(face_oval)

    file = request.files['image'].read()  ## byte file
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    ######### Do preprocessing here ################
    # img[img > 150] = 0
    ## any random stuff do here
    ################################################
    # img = Image.fromarray(img.astype("uint8"))
    # rawBytes = io.BytesIO()
    # img.save(rawBytes, "JPEG")
    # rawBytes.seek(0)
    # img_base64 = base64.b64encode(rawBytes.read())
    ans = facePoints(FaceLandMarks(image=img).face_mesh())
    ans.left_eye()
    jans = {}
    if face_oval:
        jans["face_oval"] = ans.face_oval().tolist()
    if left_eye:
        jans["left_eye"] = ans.left_eye().tolist()
    if right_eye:
        jans["right_eye"] = ans.right_eye().tolist()
    if left_eye_brow:
        jans["left_eye_brow"] = ans.left_eye_brow().tolist()
    if right_eye_brow:
        jans["right_eye_brow"] = ans.right_eye_brow().tolist()

    return jsonify({'Points': jans})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
