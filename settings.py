import os

import cv2

from utils import NumpyEncoder

DEBUG = False

BASE_PATH = os.path.dirname(__file__)
MODEL_WEIGHTS_PATH = os.path.join(BASE_PATH, 'models/keras_facenet_weights.h5')
CLASSIFIER_PATH = os.path.join(BASE_PATH, 'models/knn_classifier2.pkl')
HAARCASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

RESTFUL_JSON = {'cls': NumpyEncoder}
JWT_SECRET_KEY = 'dQeyjj5YvqCRdP6C8EH47FTmKDg6yPE0'
JWT_VERIFY_EXPIRATION = False
