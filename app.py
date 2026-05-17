from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import math
import base64
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)

detector = HandDetector(maxHands=1)

classifier = Classifier(
    "Model/keras_model.h5",
    "Model/labels.txt"
)

labels = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

offset = 20
imgSize = 300

word = ""
current_letter = ""


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    global current_letter

    try:

        data = request.json['image']

        encoded_data = data.split(',')[1]

        nparr = np.frombuffer(
            base64.b64decode(encoded_data),
            np.uint8
        )

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        hands, img = detector.findHands(img)

        if hands:

            hand = hands[0]

            x, y, w, h = hand['bbox']

            imgWhite = np.ones(
                (imgSize, imgSize, 3),
                np.uint8
            ) * 255

            imgCrop = img[
                y-offset:y+h+offset,
                x-offset:x+w+offset
            ]

            aspectRatio = h / w

            if aspectRatio > 1:

                k = imgSize / h

                wCal = math.ceil(k * w)

                imgResize = cv2.resize(
                    imgCrop,
                    (wCal, imgSize)
                )

                wGap = math.ceil(
                    (imgSize - wCal) / 2
                )

                imgWhite[:, wGap:wCal+wGap] = imgResize

            else:

                k = imgSize / w

                hCal = math.ceil(k * h)

                imgResize = cv2.resize(
                    imgCrop,
                    (imgSize, hCal)
                )

                hGap = math.ceil(
                    (imgSize - hCal) / 2
                )

                imgWhite[hGap:hCal+hGap, :] = imgResize

            prediction, index = classifier.getPrediction(
                imgWhite,
                draw=False
            )

            current_letter = labels[index]

            return jsonify({
                "success": True,
                "letter": current_letter,
                "confidence": float(prediction[index])
            })

        return jsonify({
            "success": False,
            "letter": "",
            "confidence": 0
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route('/add_letter', methods=['POST'])
def add_letter():

    global word, current_letter

    if current_letter:
        word += current_letter

    return jsonify({
        "word": word
    })


@app.route('/undo_letter', methods=['POST'])
def undo_letter():

    global word

    if len(word) > 0:
        word = word[:-1]

    return jsonify({
        "word": word
    })


@app.route('/reset', methods=['POST'])
def reset():

    global word

    word = ""

    return jsonify({
        "word": word
    })


@app.route('/get_word')
def get_word():

    return jsonify({
        "word": word
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
