import enum
from logging import exception
from operator import index
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------- GLOBALS --------------------------------- #

WEIGHTS_FILES = ['sequential_dense_MatMul.csv']
RAW_IMAGE_FILE = r'E:\Marijan\Faks\8. semestar\PDS\Seminar izvještaj\MNIST_OCR\img3.bmp'
# RAW_IMAGE_FILE = r'raw_image_bytes.txt'
IMAGE_SHAPE = (28, 28)
# LAYERS = [784,10]
M = 0.0010791111271828413 / 2.446028232574463
OUTPUT_OFFSET = 22

# --------------------------------- FUNCTIONS -------------------------------- #

def ReadImageBytes(path : str, dtype = np.float64) -> np.array:
    data = ''
    with open(path, 'r') as file:
        data = file.read()
    data = data.split(',')
    data = np.array([float(x) for x in data], dtype=dtype)
    data_size = data.shape[0]
    data = data.reshape(1, data_size)
    print(data)
    return data


def ReadImage(path : str, dtype = np.float64) -> np.array:
    data = Image.open(path)
    data = np.array(data, dtype=dtype)
    data *= 255
    data -= 128
    data_size = data.shape[0] * data.shape[1]
    data = data.reshape(1, data_size)
    print(data)
    return data


def ShowImage(image : np.array, shape : tuple):
    data = image.reshape(shape)
    print(data)
    plt.imshow(data)
    plt.show()


def ReadWeights(file_names : Tuple[str], dtype = np.float64) -> List[np.array]:
    weights = []
    for weights_file in file_names:
        current_layer_weight_matrix = np.genfromtxt(weights_file, delimiter=',', dtype=dtype)
        if len(current_layer_weight_matrix.shape) == 1:
            current_layer_weight_matrix = current_layer_weight_matrix.reshape((1, current_layer_weight_matrix.shape[0]))
        current_layer_weight_matrix = current_layer_weight_matrix.transpose()
        weights.append(current_layer_weight_matrix)
    return weights

def MultiplyMatrices(_input : np.array, _weights : np.array, fnAdd, fnMpy, fnActivation, fnCast) -> np.array:
    assert isinstance(_input, np.ndarray)
    assert isinstance(_weights, np.ndarray)

    input_rows, input_cols = _input.shape
    weights_rows, weights_cols = _weights.shape

    if input_cols != weights_rows:
        raise ValueError(f'Invalid matrix shapes for multiplication. INPUT = {_input.shape}, WEIGHTS = {_weights.shape}')
    
    _output = np.zeros((input_rows, weights_cols)).astype(np.int32)

    for i in range(0, input_rows):
        for j in range(0, weights_cols):
            for k in range(0, input_cols): # or 0, weigths_rows
                mpy_result = fnMpy(_input[i][k], _weights[k][j])
                add_result = fnAdd(_output[i][j], mpy_result)
                _output[i][j] = add_result
            _output[i][j] = fnCast(_output[i][j])
        

    vecActivation = np.vectorize(fnActivation)
    _output = vecActivation(_output)
    
    return _output

# ----------------------------------- CODE ----------------------------------- #

image = ReadImage(RAW_IMAGE_FILE, dtype=np.int32)
# image = ReadImageBytes(RAW_IMAGE_FILE)
ShowImage(image, IMAGE_SHAPE)
weight_matrices = ReadWeights(WEIGHTS_FILES, dtype=np.int32)
# print(weight_matrices)
print(weight_matrices[0].shape)

FN_CLIP = lambda x, lower, upper: upper if x > upper else (lower if x < lower else x)

def FN_MPY_INT8(a, b):
    sign_a = 1 if a > 0 else 0
    sign_b = 1 if b > 0 else 0

    a = abs(a) & 0x7F
    b = abs(b) & 0x7F

    c = a * b
    c = c & 0x7F

    sign = sign_a ^ sign_b

    c = ((-1)**sign) * c

    return c

def FN_ADD_INT8(a, b):
    c = a + b
    if c > 127:
        c = -128 + (c - 128)
    elif c < -128:
        c = 128 + (c + 128)
    return c


def FACTORY_FN_MPY_QUANT(value_offset : int,weight_offset : int):
    return lambda a,b: (int(a) + value_offset) * (int(b) + weight_offset)

def FACTORY_FN_CLAMP(M : float, output_offset : int):
    return lambda x : FN_CLIP(float(x) * M + output_offset, -128, 127)

FN_ADD = lambda a,b: int(a) + int(b)
FN_ADD_MOD_256_SIGNED = lambda a,b: (int(a) + int(b)) % 256 - 128
FN_ADD_MOD_256_SIGNED_wPREOFFSET = lambda a,b: (a + 128 + b + 128) % 256 - 128
FN_ADD_MOD_256 = lambda a,b: (int(a) + int(b)) % 256
FN_ADD_0xff = lambda a,b : (int(a) + int(b)) & 0xFF
FN_MPY_0xff = lambda a,b : (int(a) * int(b)) & 0xFF
FN_ADD_CLIP_SI = lambda a,b :  FN_CLIP(int(a) + int(b), -128, 127)
FN_MPY_CLIP_SI = lambda a,b :  FN_CLIP(int(a) * int(b), -128, 127)
FN_MPY_MOD_256_SIGNED = lambda a,b: (int(a) * int(b)) % 256 - 128
FN_MPY_MOD_256_SIGNED_wPREOFFSET = lambda a,b: ((int(a) + 128) * (int(b) + 128)) % 256 - 128
FN_MPY_MOD_256 = lambda a,b: (int(a) * int(b)) % 256
FN_MPY = lambda a,b: int(a) * int(b)
FN_MPY_DEQUANTIZED = lambda a,b: int(a) * (int(b) - 5)
FN_RELU = lambda x: 0 if x < 0 else x
FN_IDENTITY = lambda x : x
FN_MOD_256 = lambda x : x % 256
# FN_0xff = lambda x : (x & 0xFF)
FN_MOD_256_SIGNED = lambda x : x % 256 - 128
FN_DEQUANTIZE = lambda x : 2.527824476961375679123183655418e-4 * x


_output = []
for layer, weights in enumerate(weight_matrices):
    if layer == 0:
        _input = image
    else:
        _input = _output

    _output = MultiplyMatrices(
        _input, 
        weights,
        fnAdd=FN_ADD,
        fnMpy=FACTORY_FN_MPY_QUANT(value_offset=0, weight_offset=0),
        # fnMpy=FN_MPY,
        fnActivation=FN_IDENTITY,
        fnCast=FACTORY_FN_CLAMP(M=M, output_offset=OUTPUT_OFFSET)
        # fnCast=FN_IDENTITY
    )

    print(_output)

_output = _output[0]
print(f'PREDICTION = {_output.tolist().index(max(_output))}')

