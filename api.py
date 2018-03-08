from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage

from face import FaceExtractor
from image_preprocessing import load_image
from model.inception_resnet_v1 import InceptionResNetV1
from settings import DEBUG, MODEL_WEIGHTS_PATH
from utils import NumpyEncoder

app = Flask(__name__)
api = Api(app)
app.debug = DEBUG
model = None
face_extractor = None


class Config:
    RESTFUL_JSON = {'cls': NumpyEncoder}


class RecognizeFace(Resource):
    def post(self):
        global model
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        _image = args['image']
        image = load_image(_image, 160)
        return {
            'image': _image.filename,
            'embeddings': model.predict(image)[0]
        }


api.add_resource(RecognizeFace, '/recognize-face')

if __name__ == '__main__':
    app.config.from_object(Config)

    face_extractor = FaceExtractor()
    print('Loading Face Recognition model...')
    model = InceptionResNetV1()
    model.load_weights(MODEL_WEIGHTS_PATH)
    print('Face Recognition model is loaded')
    app.run(use_reloader=False)
