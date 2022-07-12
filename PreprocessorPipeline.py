from typing import Tuple
import numpy
from PIL import Image, ImageOps

class Filter:
    def __init__(self, callback) -> None:
        self._callback = callback

    def Execute(self, input):
        return self._callback(input)

class FilterPipeline:
    def __init__(self, filters : Tuple[Filter]) -> None:
        self._filters = filters
    
    def Execute(self, input):
        currentInput = input
        for filter in self._filters:
            filterOutput = filter.Execute(currentInput)
            currentInput = filterOutput
        return filterOutput

def ByteNormalizationFilter():
    return Filter(lambda x : x / 255.0)

def ResizeFilter(size):
    return Filter(lambda x : x.resize(size, Image.ANTIALIAS))

def FlattenFilter():
    return Filter(lambda x : numpy.reshape(x, -1, 'C'))

def ToNdarrayFilter(dtype = numpy.float32):
    return Filter(lambda x : numpy.array(x, dtype=dtype))

def GrayscaleFilter():
    return Filter(lambda x : ImageOps.grayscale(x))

def InvertImageFilter():
    return Filter(lambda x : ImageOps.invert(x))

def CastAs(dtype):
    return Filter(lambda x : x.astype(dtype))

def Offset(offset : int):
    return Filter(lambda x : x - offset)

def _SaveImageAndReturnIt(path : str, image):
    assert isinstance(image, numpy.ndarray)
    image_cpy = [str(x) for x in image]
    image_cpy = ','.join(image_cpy)
    with open(path, 'w') as file:
        file.write(image_cpy)
    return image

def ExportRaw(path : str):
    return Filter(lambda x : _SaveImageAndReturnIt(path, x))

def _SaveImageBytesAndReturnIt(path : str, image):
    assert isinstance(image, numpy.ndarray)
    image = image.view(numpy.int8)
    with open(path, 'wb') as file:
        for b in image:
            file.write(b)
    return image

def ExportBytes(path : str):
    return Filter(lambda x : _SaveImageBytesAndReturnIt(path, x))
