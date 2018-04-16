import numpy as np
from flask import Flask, request
from flask_jwt import JWT, jwt_required
from flask_restful import Resource, Api, reqparse
from flask_sockets import Sockets
from passlib.hash import pbkdf2_sha256
from scipy.misc import imread
from scipy.spatial import distance
from werkzeug.datastructures import FileStorage

from classification import KNeighborsClassifier
from face import FaceExtractor
from image_preprocessing import load_image, prewhiten
from models.inception_resnet_v1 import InceptionResNetV1

app = Flask(__name__)
api = Api(app)
app.debug = DEBUG
model = None
face_extractor = None
classifier = None

User = namedtuple('User', 'id, username, password')
with open('users.json', encoding='utf-8') as f:
    users = [User(user['id'], user['username'], user['password']) for user in json.load(f)]

username_table = {u.username: u for u in users}
userid_table = {u.id: u for u in users}


def authenticate(username, password):
    user = username_table.get(username)
    if user and pbkdf2_sha256.verify(user.password, password):
        return user


def identify(payload):
    return userid_table.get(payload['identity'])


jwt = JWT(app, authenticate, identify)


class RecognizeFace(Resource):
    @jwt_required()
    def post(self):
        global model
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        _image = args['image']
        image = load_image(_image, 160)
        return {
            'image': _image.filename,
            'embeddings': model.predict(prewhiten(image))[0]
        }


class FindFaces(Resource):
    @jwt_required()
    def post(self):
        global face_extractor
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        _image = args['image']
        image = imread(_image, mode='RGB')
        faces = face_extractor.find_faces(image)
        return {
            'image': _image.filename,
            'found_faces': len(faces),
            'faces': faces
        }


class RecognizeFaces(Resource):
    @jwt_required()
    def post(self):
        global face_extractor
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        _image = args['image']
        image = imread(_image, mode='RGB')
        faces = face_extractor.extract_faces(image, image_size=160, margin=0.1)
        input_tensor = np.array([prewhiten(face[0]) for face in faces])
        predictions = model.predict(input_tensor)
        return {
            'image': _image.filename,
            'found_faces': len(faces),
            'faces': [{
                'area': face[1],
                'embeddings': prediction
            } for face, prediction in zip(faces, predictions)],
        }


class CompareEmbeddings(Resource):
    @jwt_required()
    def post(self):
        face_embeddings = request.get_json()
        _distance = distance.euclidean(*face_embeddings)
        return {
            'distance': _distance,
        }


class CompareFaces(Resource):
    @jwt_required()
    def post(self):
        global face_extractor, model
        parse = reqparse.RequestParser()
        parse.add_argument('image1', type=FileStorage, location='files')
        parse.add_argument('image2', type=FileStorage, location='files')
        args = parse.parse_args()
        image1, image2 = args['image1'], args['image2']
        image1_faces = face_extractor.extract_faces(
            imread(image1, mode='RGB'),
            image_size=160
        )
        image2_faces = face_extractor.extract_faces(
            imread(image2, mode='RGB'),
            image_size=160
        )

        result = {
            'info': [
                {
                    'image': image1.filename,
                    'found_faces': len(image1_faces),
                    'faces': [{
                        'area': face[1]
                    } for face in image1_faces]
                },
                {
                    'image': image2.filename,
                    'found_faces': len(image2_faces),
                    'faces': [{
                        'area': face[1]
                    } for face in image2_faces]
                }
            ]
        }

        if len(image1_faces) > 1:
            return {
                'error': f'Found {len(image1_faces)} faces on first image',
                'distance': None,
                **result
            }
        if len(image2_faces) > 1:
            return {
                'error': f'Found {len(image2_faces)} faces on second image',
                'distance': None,
                **result
            }
        face1_input_tensor = np.array([prewhiten(image1_faces[0][0])])
        face1_embeddings = [model.predict(face1_input_tensor)[0]]

        face2_input_tensor = np.array([prewhiten(image2_faces[0][0])])
        face2_embeddings = model.predict(face2_input_tensor)[0]

        _distance = distance.euclidean(face1_embeddings, face2_embeddings)

        return {
            'distance': _distance,
            **result
        }


api.add_resource(FindFaces, '/find-faces')
api.add_resource(CompareEmbeddings, '/compare-embeddings')
api.add_resource(CompareFaces, '/compare-faces')
api.add_resource(RecognizeFace, '/recognize-face')
api.add_resource(RecognizeFaces, '/recognize-faces')

if __name__ == '__main__':
    app.config.from_pyfile('settings.py')
    face_extractor = FaceExtractor()
    print('Loading Face Recognition model...')
    model = InceptionResNetV1()
    model.load_weights(app.config['MODEL_WEIGHTS_PATH'])
    print('Face Recognition model is loaded')
    print('Loading Classifier model...')
    classifier_path = app.config.get('CLASSIFIER_PATH')
    if classifier_path:
        classifier: KNeighborsClassifier = KNeighborsClassifier.from_file(classifier_path)
    else:
        classifier = KNeighborsClassifier()
    print('Classifier model is loaded')
    app.run(use_reloader=False, port=5001)
