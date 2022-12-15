from flask import Flask, jsonify, request
import tensorflow as tf
from io import BytesIO
from PIL import Image
import requests
from keras.utils import img_to_array

app = Flask(__name__)
model = tf.keras.models.load_model('models/kitachi.h5')
classes = ['garpu besi', 'garpu plastik', 'mangkok kaca', 'mangkok plastik', 'panci besi', 'panci listrik', 'piring kaca', 'piring plastik', 'sendok besi', 'sendok nasi', 'sendok plastik', 'sumpit kayu', 'sumpit plastik', 'wajan besi']


# load image from url for prediction (input shape is 150x150x3)
def load_img(url):
    response = requests.get(url)
    img_bytes = BytesIO(response.content)
    img = Image.open(img_bytes)
    img = img.convert('RGB')
    img = img.resize((150, 150), Image.NEAREST)
    img = img_to_array(img)
    return img


@app.route('/')
def home():
    return '<center style="font-size:40px;margin-top:23%">Kitachi AI Service 👋</center>'


@app.route('/api/v1/ai', methods=['POST'])
def ai():
    try:
        image_url = request.form['url']
        img = load_img(image_url)
        img = img.reshape(1, 150, 150, 3)
        img = img.astype('float32')
        img = img / 255.0
        prediction = model.predict(img)
        prediction = prediction[0]
        result = []
        for i in range(len(prediction)):
            result.append({'label': classes[i], 'score': float(prediction[i])})
        result = sorted(result, key=lambda x: x['score'], reverse=True)
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        print(e)
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


if __name__ == '__main__':
    app.run()
