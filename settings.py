import os

import cv2

DEBUG = False

BASE_PATH = os.path.dirname(__file__)
MODEL_WEIGHTS_PATH = os.path.join(BASE_PATH, 'models/keras_facenet_weights.h5')
HAARCASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
