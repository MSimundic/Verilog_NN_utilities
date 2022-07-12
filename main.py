# ---------------------------------------------------------------------------- #
#                              DATA PREPROCESSING                              #
# ---------------------------------------------------------------------------- #

# ------------------------------- WEIGHTS - CSV ------------------------------ #

# import csv

# def ReadCSV(filepath : str) -> list:
#     data = []
#     with open(filepath, 'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             data += [int(x) for x in row]
#     return data


# INPUT_FILES = ['sequential_dense_MatMul.csv']

# DATA = []

# for file in INPUT_FILES:
#     DATA += ReadCSV(file)


# ----------------------------------- IMAGE ---------------------------------- #


FILE_PATH = r'raw_image_bytes.txt'

contents = open(FILE_PATH).read()
image_bytes = contents.split(',')
DATA = [int(x) for x in image_bytes]



# ---------------------------------------------------------------------------- #
#                               INPUT PARAMETERS                               #
# ---------------------------------------------------------------------------- #

DATA_BIT_WIDTH = 8
ADDRESS_BIT_WIDTH = 11
# DATA = [784,10]
# PADDING_VALUE = 2 ** DATA_BIT_WIDTH - 1
PADDING_VALUE = 0
OUTPUT_FILE_NAME = 'neurons.txt'

# --------------------- DEFINED TRANSFORMATION FUNCTIONS --------------------- #

fnNumToTwosComplement = lambda x : x if x >= 0 else 2**DATA_BIT_WIDTH + x
fnNumToTwoDigitHex = lambda x : hex(x).replace('0x', '').upper()
fnNumToTwoBin = lambda x : bin(x).replace('0b', '')
fnNumToStr = lambda x : str(x)

# ---------------- ORDER OF APPLICATION OF TRANSFORMATION FNs ---------------- #

DATA_TRANSFORMATIONS = [
    fnNumToTwosComplement,
    fnNumToTwoBin
    # fnNumToTwoDigitHex
    # fnNumToStr
]


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

# --------------------------------- VARIABLES -------------------------------- #

number_of_data = 2**ADDRESS_BIT_WIDTH
data_padding_count = number_of_data - len(DATA)

# ----------------------------------- MAIN ----------------------------------- #

# data = DATA
data = ReturnPaddedCopy(DATA, PADDING_VALUE, data_padding_count)
data = ApplyAllTransformations(data, DATA_TRANSFORMATIONS)
WriteListToFile(data, OUTPUT_FILE_NAME)

print(data)



