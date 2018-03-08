from typing import Union

import cv2
import numpy as np
from numpy import ndarray as array

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
