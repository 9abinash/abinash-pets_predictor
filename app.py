import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import io
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import request
import warnings
warnings.filterwarnings("ignore")
import keras
from keras.backend import clear_session

import pickle
import tensorflow as tf
graph = tf.get_default_graph()


app = Flask(__name__)
# model = pickle.load(open('mobile_abi.h5', 'rb'))

model=load_model('mobile_abi.h5')

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():
    # '''
    # For rendering results on HTML GUI
    # '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='Employee Salary should be  ')

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    img_array_expanded_dims = np.expand_dims(image, axis=0)

    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# def prepare_image(file):
    # img_path = '/Users/abigiri/Desktop/'
    # img = image.load_img(img_path + file, target_size=(224, 224))
    # img_array = image.img_to_array(img)
    # img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    # return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

output=''
@app.route("/predict", methods=["POST"])
def predict():
    image = flask.request.files.get('imagefile', '')
    image = image.read()
    image = Image.open(io.BytesIO(image))
    processed_image = preprocess_image(image, target_size=(224, 224))
    with graph.as_default():
        prediction = model.predict(processed_image)
    if prediction[0][0] > prediction[0][1]:
        text='Congrats!! you are a cat person.'
    else:
        text='Congrats!! you are a dog person. Kevin the dog likes you <3'
    return render_template('index.html', prediction_text=text )
if __name__ == "__main__":
    app.run(debug=True)