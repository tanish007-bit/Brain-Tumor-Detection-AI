import os
import tensorflow as tf
import numpy as np

from PIL import Image
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained model
model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Class names
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


# Prediction function
def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)

    input_img = np.expand_dims(image, axis=0)

    prediction = model.predict(input_img)
    result = np.argmax(prediction, axis=1)

    return result[0]


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']

        basepath = os.path.dirname(__file__)

        upload_path = os.path.join(basepath, 'uploads')

        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        value = getResult(file_path)

        result = get_className(value)

        return result


if __name__ == '__main__':
    app.run(debug=True)