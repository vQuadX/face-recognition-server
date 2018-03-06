import os

from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage

from image_preprocessing import load_image
from model.inception_resnet_v1 import InceptionResNetV1

BASE_PATH = os.path.dirname(__file__)

app = Flask(__name__)
api = Api(app)
model = None


class RecognizeFace(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        _image = args['image']
        image = load_image(_image, 160)
        return {
            'image': _image.filename,
            'embeddings': model.predict(image)[0].tolist()
        }


api.add_resource(RecognizeFace, '/recognize-face')

if __name__ == '__main__':
    print('Loading Face Recognition model...')
    model = InceptionResNetV1()
    model.load_weights(os.path.join(BASE_PATH, os.environ.get('MODEL_WEIGHTS_PATH')))
    print('Face Recognition model is loaded')
    app.run(use_reloader=False)
