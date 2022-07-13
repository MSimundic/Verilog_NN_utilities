
# ---------------------------------------------------------------------------- #
#                              DATA PREPROCESSING                              #
# ---------------------------------------------------------------------------- #

# ------------------------------- WEIGHTS - CSV ------------------------------ #

import csv
from typing import List

def ReadCSV(filepath : str) -> list:
    data = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data += [int(x) for x in row]
    return data


# ----------------------------------- IMAGE ---------------------------------- #


def ReadImage(file_path : str) -> List[int]:
    contents = open(file_path).read()
    image_bytes = contents.split(',')
    data = [int(x) for x in image_bytes]
    return data


# --------------------- DEFINED TRANSFORMATION FUNCTIONS --------------------- #

def factory_fnNumToTwosComplement(bit_width : int):
    return lambda x : x if x >= 0 else 2**bit_width + x

fnNumToTwoDigitHex = lambda x : hex(x).replace('0x', '').upper()
fnNumToBin = lambda x : bin(x).replace('0b', '')
fnNumToStr = lambda x : str(x)


# ---------------------------------------------------------------------------- #
#                                     CODE                                     #
# ---------------------------------------------------------------------------- #


# --------------------------------- FUNCTIONS -------------------------------- #

def ReturnPaddedCopy(data : list, pad_value, pad_count) -> list:
    return data + [pad_value] * pad_count

def ApplyTransformation(data : list, fnTransform) -> list:
    return [fnTransform(x) for x in data]

def ApplyAllTransformations(data : list, transformations : list) -> list:
    for transform in transformations:
        data = ApplyTransformation(data, transform)
    return data

def WriteListToFile(data : list, filepath : str):
    with open(filepath, 'w') as file:
        file.write('\n'.join(data))

# ---------------------------------------------------------------------------- #
#                               HELPER FUNCTIONS                               #
# ---------------------------------------------------------------------------- #

# ---------------------------------- WEIGHTS --------------------------------- #


def ParseWeights(
    weight_files : List[str],
    weight_data_bus_width : int,
    weight_address_bus_width : int,
    output_file_path : str
    ):

    weights = []

    for file in weight_files:
        weights += ReadCSV(file)
        
    padding_value = 0
    number_of_data = 2**weight_address_bus_width
    data_padding_count = number_of_data - len(weights)

    data_transformations = [
        factory_fnNumToTwosComplement(weight_data_bus_width),
        fnNumToBin
    ]

    data = ReturnPaddedCopy(weights, padding_value, data_padding_count)
    data = ApplyAllTransformations(data, data_transformations)
    WriteListToFile(data, output_file_path)
    print(data)

# ----------------------------------- IMAGE ---------------------------------- #

# DEPRECATED, USED ONLY FOR DEBBUGGING IN EARLY STAGES

def ParseImage(
    image_txt_bytes_path : str,
    neuron_data_bus_width : int,
    neuron_address_bus_width : int,
    output_file_path : str
    ):

    neurons = ReadImage(image_txt_bytes_path)

    padding_value = 0
    number_of_data = 2**neuron_address_bus_width
    data_padding_count = number_of_data - len(neurons)

    data_transformations = [
        factory_fnNumToTwosComplement(neuron_data_bus_width),
        fnNumToBin
    ]

    data = ReturnPaddedCopy(neurons, padding_value, data_padding_count)
    data = ApplyAllTransformations(data, data_transformations)
    WriteListToFile(data, output_file_path)
    print(data)

# ---------------------------------- LAYERS ---------------------------------- #

def ParseLayers(
    layer_sizes : List[int],
    ip_data_bus_width : int,
    ip_address_bus_width : int,
    output_file_path : str
    ):

    padding_value = (2 ** ip_data_bus_width) - 1 # ALL ONES
    number_of_data = 2**ip_address_bus_width
    data_padding_count = number_of_data - len(layer_sizes)

    data_transformations = [
        fnNumToTwoDigitHex
    ]

    data = ReturnPaddedCopy(layer_sizes, padding_value, data_padding_count)
    data = ApplyAllTransformations(data, data_transformations)
    WriteListToFile(data, output_file_path)
    print(data)


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

ParseWeights(
    weight_files=['sequential_dense_MatMul.csv'],
    weight_data_bus_width=8,
    weight_address_bus_width=13,
    output_file_path='weights_out.txt'
)

ParseLayers(
    layer_sizes=[784, 10],
    ip_data_bus_width=16,
    ip_address_bus_width=8,
    output_file_path='instructions_out.txt'
)