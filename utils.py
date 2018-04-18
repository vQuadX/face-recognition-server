from json import JSONEncoder

import numpy as np


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def serialize_area(area):
    return {
        'x': int(area[0]),
        'y': int(area[1]),
        'width': int(area[2]),
        'height': int(area[3])
    }
