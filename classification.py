import joblib
import numpy as np


class KNeighborsClassifier:
    def __init__(self, n_neighbors: int = 1):
        self.n_neighbors = n_neighbors

        self.classes = {}
        self.__uuid_classes: np.ndarray = None

        self._last_class_id = 0
        self._X: np.ndarray = None
        self._y: np.ndarray = None

    @classmethod
    def from_file(cls, file_name):
        return joblib.load(file_name)

    def partial_fit(self, X, y):
        _y = np.empty(y.shape, dtype=np.int)
        for i, cls in enumerate(y):
            class_id = self.classes.setdefault(cls, self._last_class_id)
            _y[i] = class_id
            if class_id == self._last_class_id:
                if self.__uuid_classes is None:
                    self.__uuid_classes = np.array([cls], dtype='S36')
                else:
                    self.__uuid_classes = self._extend(self.__uuid_classes, [cls])
                self._last_class_id += 1
        if self._y is None:
            self._y = np.array(_y)
        else:
            self._y = self._extend(self._y, _y)
        if self._X is None:
            self._X = np.array(X)
        else:
            self._X = self._extend(self._X, X)

    def __calc_distances(self, X):
        distances = np.empty(self._y.shape, dtype=np.float)

        for i in range(self._y.shape[0]):
            distances[i] = self.distance(self._X[i], X)

        return distances

    def predict(self, X):
        return self._predict(X)

    def predict_on_batch(self, X):
        distances = np.empty(X.shape[0], dtype=np.float32)
        uuidx = np.empty(X.shape[0], dtype='S36')

        for i, x in enumerate(X):
            distances[i], uuidx[i] = self._predict(x)

        return distances, uuidx

    def _predict(self, X):
        distances = self.__calc_distances(X)

        if self.n_neighbors == 1:
            min_idx = np.argmin(distances)
            predicted_class_id = self._y[min_idx]
            distance = distances[min_idx]
        else:
            min_idx = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            y_min = self._y[min_idx]
            cls, counts = np.unique(y_min, return_counts=True)
            argmax = np.argmax(counts)
            predicted_class_id = cls[argmax]
            distance = distances[predicted_class_id]

        return distance, self.__uuid_classes[predicted_class_id]

    def kneighbors(self, X, n_neighbors=None):
        distances = self.__calc_distances(X)
        if n_neighbors:
            min_idx = np.argpartition(distances, n_neighbors)[:n_neighbors]
            neighbors = sorted(zip(distances[min_idx], min_idx), key=lambda x: x[0])
            neighbors_distances = np.fromiter((n[0] for n in neighbors), dtype=np.float64)
            neighbors_classes = self._y[[n[1] for n in neighbors]]
            neighbors_uuid = self.__uuid_classes[neighbors_classes]
        else:
            idx = np.argsort(distances)
            neighbors_distances = distances[idx]
            neighbors_classes = self._y[idx]
            neighbors_uuid = self.__uuid_classes[neighbors_classes]

        return neighbors_distances, neighbors_uuid

    def save(self, file_name='knn_classifier.pkl'):
        joblib.dump(self, file_name)

    @staticmethod
    def _extend(arr, values):
        shape = arr.shape
        new_shape = shape[0] + len(values), *shape[1:]
        x = np.resize(arr, new_shape)
        for i in range(shape[0], new_shape[0]):
            if len(shape) == 2:
                x[i, :] = values[shape[0] - i]
            else:
                x[i] = values[shape[0] - i]
        return x

    @staticmethod
    def distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))
