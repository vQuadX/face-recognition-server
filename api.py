import json
import os
from collections import namedtuple
from io import BytesIO
from itertools import zip_longest
from json import JSONDecodeError
from os import listdir
from uuid import uuid4

import numpy as np
from flask import Flask, request, send_from_directory
from flask_jwt_simple import JWTManager, jwt_required, create_jwt
from flask_restful import Resource, Api, reqparse
from flask_sockets import Sockets
from imageio import imsave
from passlib.hash import pbkdf2_sha256
from scipy.misc import imread
from scipy.spatial import distance
from werkzeug.datastructures import FileStorage

from classification import KNeighborsClassifier
from face import FaceExtractor
from image_preprocessing import load_image, prewhiten
from models.inception_resnet_v1 import InceptionResNetV1
from utils import serialize_area

app = Flask(__name__)
api = Api(app)
jwt = JWTManager(app)
sockets = Sockets(app)

model = None
face_extractor = None
classifier = None

User = namedtuple('User', 'id, username, password')
with open('users.json', encoding='utf-8') as f:
    users = [User(user['id'], user['username'], user['password']) for user in json.load(f)]

username_table = {u.username: u for u in users}
userid_table = {u.id: u for u in users}


class Auth(Resource):
    def post(self):
        if not request.is_json:
            return {'msg': 'Missing JSON in request'}, 400

        params = request.get_json()
        username = params.get('username', None)
        password = params.get('password', None)

        if not username:
            return {'msg': 'Missing username parameter'}, 400
        if not password:
            return {'msg': 'Missing password parameter'}, 400

        user = username_table.get(username)
        if user and pbkdf2_sha256.verify(user.password, password):
            return {'jwt': create_jwt(identity=username)}
        else:
            return {'msg': 'Bad username or password'}, 401


class RecognizeFace(Resource):
    @jwt_required
    def post(self):
        global model
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        _image = args['image']
        image = load_image(_image, 160)
        return {
            'embeddings': model.predict(image)[0]
        }


class FindFaces(Resource):
    @jwt_required
    def post(self):
        global face_extractor
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        _image = args['image']
        image = imread(_image, mode='RGB')
        faces = face_extractor.find_faces(image)
        return {
            'found_faces': len(faces),
            'faces': [serialize_area(face_area) for face_area in faces]
        }


class RecognizeFaces(Resource):
    @jwt_required
    def post(self):
        global face_extractor
        parse = reqparse.RequestParser()
        parse.add_argument('image', type=FileStorage, location='files')
        args = parse.parse_args()
        _image = args['image']
        image = imread(_image, mode='RGB')
        faces = face_extractor.extract_faces(image, image_size=160, margin=0.1)
        if len(faces):
            input_tensor = np.array([prewhiten(face[0]) for face in faces])
            predictions = model.predict(input_tensor)
            return {
                'found_faces': len(faces),
                'faces': [{
                    'area': serialize_area(face_area),
                    'embeddings': prediction
                } for face_area, prediction in zip_longest((face[1] for face in faces), predictions)],
            }
        else:
            return {
                'found_faces': 0,
                'faces': []
            }


class CompareEmbeddings(Resource):
    @jwt_required
    def post(self):
        face_embeddings = request.get_json()
        _distance = distance.euclidean(*face_embeddings)
        return {
            'distance': _distance,
        }


class CompareFaces(Resource):
    @jwt_required
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
                    'found_faces': len(image1_faces),
                    'faces': [{
                        'area': serialize_area(face_area)
                    } for face_area in (face[1] for face in image1_faces)]
                },
                {
                    'found_faces': len(image2_faces),
                    'faces': [serialize_area(face_area) for face_area in (face[1] for face in image2_faces)]
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
        face1_embeddings = model.predict(face1_input_tensor)[0]

        face2_input_tensor = np.array([prewhiten(image2_faces[0][0])])
        face2_embeddings = model.predict(face2_input_tensor)[0]

        _distance = distance.euclidean(face1_embeddings, face2_embeddings)

        return {
            'distance': _distance,
            **result
        }


class IdentifyFaces(Resource):
    @jwt_required
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', type=FileStorage, location='files')
        args = parser.parse_args()

        face_image = args['image']
        faces = face_extractor.extract_faces(
            imread(face_image, mode='RGB'),
            image_size=160,
            margin=0.1
        )
        if len(faces):
            input_tensor = np.array([prewhiten(face[0]) for face in faces])
            faces_embeddings = model.predict(input_tensor)
            distances, identifiers = classifier.predict_on_batch(faces_embeddings)
            return {
                'found_faces': len(faces),
                'persons': [{
                    'area': serialize_area(face_area),
                    'id': uuid.decode('utf-8') if uuid and dist is not None and dist <= 0.6 else None,
                    'distance': float(dist) if dist is not None else None
                } for face_area, dist, uuid in zip_longest((face[1] for face in faces), distances, identifiers)],
            }
        else:
            return {
                'found_faces': 0,
                'persons': []
            }


class AddPerson(Resource):
    @jwt_required
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', type=FileStorage, location='files')
        args = parser.parse_args()

        image = args.get('image')
        image = imread(image, mode='RGB')
        faces = face_extractor.extract_faces(
            image,
            image_size=160,
            margin=0.1
        )
        if not len(faces):
            return {
                'error': 'Faces not found on image'
            }
        elif len(faces) > 1:
            return {
                'error': f'Found more than 1 face ({len(faces)})'
            }

        face_image = faces[0][0]
        face_input_tensor = np.array([prewhiten(face_image)])
        face_embeddings = model.predict(face_input_tensor)

        person_id = str(uuid4())
        class_id = classifier.partial_fit(face_embeddings, np.array([person_id]))[0]

        person_images_dir = os.path.join(app.config.get('IMAGE_DIR'), person_id)
        if not os.path.exists(person_images_dir):
            os.makedirs(person_images_dir)

        person_image = os.path.join(person_images_dir, f'{class_id}_original.jpg')
        person_face_image = os.path.join(person_images_dir, f'{class_id}.jpg')
        imsave(person_image, image)
        imsave(person_face_image, face_image)

        classifier.save('models/knn_classifier.pkl')
        return {
            'person_id': person_id
        }


class PersonImages(Resource):
    @jwt_required
    def get(self, person_id):
        person_folder = f'images/{person_id}'
        if os.path.exists(person_folder):
            return {
                'images': [f'{person_folder}/{image}' for image in listdir(person_folder) if
                           os.path.isfile(os.path.join(person_folder, image)) if 'original' not in image]
            }
        else:
            return {'error': 'Person not found'}, 400


@sockets.route('/ws')
def recognition_socket(ws):
    mode = 'face-detection'
    while not ws.closed:
        data = ws.receive()
        if not data:
            continue
        if isinstance(data, (bytes, bytearray)):
            image = BytesIO(data)
            faces = face_extractor.extract_faces(
                imread(image, mode='RGB'),
                image_size=160,
                margin=0.1
            )
            if not len(faces):
                ws.send(json.dumps({
                    'found_faces': 0,
                    'persons': []
                }))
                continue
            if mode == 'face-detection':
                ws.send(json.dumps({
                    'found_faces': len(faces),
                    'persons': [{
                        'area': serialize_area(face_area),
                    } for face_area in (face[1] for face in faces)],
                }))
            elif mode == 'face-recognition':
                input_tensor = np.array([prewhiten(face[0]) for face in faces])
                faces_embeddings = model.predict(input_tensor)
                distances, identifiers = classifier.predict_on_batch(faces_embeddings)

                ws.send(json.dumps({
                    'found_faces': len(faces),
                    'persons': [{
                        'area': serialize_area(face_area),
                        'id': uuid.decode('utf-8') if dist <= 0.6 else None,
                        'distance': float(dist)
                    } for face_area, dist, uuid in zip_longest((face[1] for face in faces), distances, identifiers)],
                }))
            else:
                ws.send(json.dumps({
                    'error': 'unexpected mode'
                }))
        else:
            try:
                data = json.loads(data)
            except JSONDecodeError:
                ws.send(json.dumps({
                    'error': 'invalid JSON'
                }))
            else:
                mode = data.get('mode')


api.add_resource(Auth, '/auth')
api.add_resource(IdentifyFaces, '/identify-faces')
api.add_resource(FindFaces, '/find-faces')
api.add_resource(CompareEmbeddings, '/compare-embeddings')
api.add_resource(CompareFaces, '/compare-faces')
api.add_resource(RecognizeFace, '/recognize-face')
api.add_resource(RecognizeFaces, '/recognize-faces')
api.add_resource(AddPerson, '/add-person')
api.add_resource(PersonImages, '/person-images/<string:person_id>')

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
        if os.path.exists(classifier_path):
            classifier: KNeighborsClassifier = KNeighborsClassifier.from_file(classifier_path)
        else:
            classifier = KNeighborsClassifier()
    else:
        classifier = KNeighborsClassifier()
    print('Classifier model is loaded')
    # app.run(use_reloader=False, port=5001)
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(('127.0.0.1', 5001), app, handler_class=WebSocketHandler)
    server.serve_forever()
