from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import math
import time
import base64
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)

# Initialize detector and classifier
detector = HandDetector(maxHands=1)

classifier = Classifier(
    "Model/keras_model.h5",
    "Model/labels.txt"
)

labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

# Image preprocessing constants
offset = 20
imgSize = 300

# State variables
current_letter = ""
confidence = 0.0
word = ""
last_letter = ""
letter_start_time = None
auto_add_delay = 5


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    global word
    global current_letter
    global confidence
    global last_letter
    global letter_start_time

    try:

        data = request.json['image']

        encoded_data = data.split(',')[1]

        nparr = np.frombuffer(
            base64.b64decode(encoded_data),
            np.uint8
        )

        img = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        imgOutput = img.copy()

        hands, img = detector.findHands(img)

        if hands:

            hand = hands[0]

            x, y, w, h = hand['bbox']

            imgWhite = np.ones(
                (imgSize, imgSize, 3),
                np.uint8
            ) * 255

            x1 = max(0, x - offset)
            y1 = max(0, y - offset)

            x2 = min(img.shape[1], x + w + offset)
            y2 = min(img.shape[0], y + h + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:

                return jsonify({
                    "success": False
                })

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

                imgWhite[:, wGap:wCal + wGap] = imgResize

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

                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(
                imgWhite,
                draw=False
            )

            current_letter = labels[index]

            confidence = prediction[index]

            # Auto add letter logic
            if current_letter == last_letter:

                if (
                    letter_start_time and
                    time.time() - letter_start_time >= auto_add_delay
                ):

                    word += current_letter

                    letter_start_time = None
                    last_letter = ""

            else:

                last_letter = current_letter
                letter_start_time = time.time()

            return jsonify({
                "success": True,
                "letter": current_letter,
                "confidence": float(confidence),
                "word": word
            })

        else:

            current_letter = ""
            confidence = 0.0
            last_letter = ""
            letter_start_time = None

            return jsonify({
                "success": False
            })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route('/add_letter', methods=['POST'])
def add_letter():

    global word
    global current_letter

    if current_letter:
        word += current_letter

    return "OK"


@app.route('/undo_letter', methods=['POST'])
def undo_letter():

    global word

    if len(word) > 0:
        word = word[:-1]

    return "OK"


@app.route('/reset', methods=['POST'])
def reset():

    global word

    word = ""

    return "OK"


@app.route('/get_word')
def get_word():

    return jsonify({
        "word": word
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
