from typing import Union, List, Tuple

import cv2
import numpy as np
from numpy import ndarray as array
from scipy.misc import imread, imresize

from image_preprocessing import reverse_channels
from settings import HAARCASCADE_PATH


class FaceExtractor:
    def __init__(self):
        self.cascade_classifier = cv2.CascadeClassifier(HAARCASCADE_PATH)

    def find_faces(self, image: Union[str, np.ndarray]) -> array:
        """Find all faces in image.

        :param image: image path or RGB image as numpy array
        :return: array or list of faces area array (x, y, width, height)
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        else:
            image = reverse_channels(image)

        faces = self.cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return faces

    def extract_faces(self, image: Union[str, np.ndarray], image_size: int = None, margin: float = None) -> List[
        Tuple[array, array]]:
        """Find and extract all faces in image.

        :param image: image path or RGB image as numpy array
        :param image_size: size of output face images
        :param margin: margin for each side (between 0 and 1)
        :return: list of (RGB face array, face area array (x, y, width, height)) tuple
        """
        if isinstance(image, str):
            image = imread(image, mode='RGB')
        faces = self.find_faces(image)

        face_images = []
        for x, y, w, h in faces:
            indent = int(min(w, h) * margin) if margin else 0
            x, y, w, h = x + indent // 2, y + indent // 2, w - indent, h - indent
            face_image = image[y: y + h, x: x + w, :]
            width, height = face_image.shape[:2]
            if image_size and (height != image_size or width != image_size):
                scaling_factor = image_size / min(width, height)
                face_image = imresize(face_image, size=(round(width * scaling_factor), round(height * scaling_factor)))
            face_images.append((face_image, np.array([x, y, w, h])))

        return face_images
