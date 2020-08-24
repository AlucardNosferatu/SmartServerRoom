import cv2
import zxing
import base64
import numpy as np
from QRCode import test_on_array
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
zx = zxing.BarCodeReader()


@app.route('/qrcode_decoder', methods=['POST'])
def qrc_decode_flask():
    if 'URL' in request.form:
        file_path = request.form['URL']
        img_array = cv2.imread(file_path)
        result = test_on_array(img_array)
    elif 'Base64' in request.form:
        b64_data = request.form['Base64']
        img_data = base64.urlsafe_b64decode(b64_data)
        np_array = np.frombuffer(img_data, dtype=np.uint8)
        img_array = cv2.imdecode(np_array, cv2.COLOR_BGR2RGB)
        result = test_on_array(img_array)
    else:
        result = False
    return jsonify({'Result': str(result)}), 201


if __name__ == '__main__':
    app.run(port='2029', debug=True)
