from copy import Error
import keras
import sys
import cv2
from matplotlib.image import imsave
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
from PreprocessorPipeline import *

required_image_shape = (28, 28)

# classes = (
#     'T-shirt/top',
#     'Trouser',
#     'Pullover',
#     'Dress',
#     'Coat',
#     'Sandal',
#     'Shirt',
#     'Sneaker',
#     'Bag',
#     'Ankle boot'
# )

def FatalError(message : str):
    print(f'ERROR: {message}')
    exit(-1)

def Predict(model, image):
    imageAsInput = numpy.array([image])
    predictions = model.predict(imageAsInput)
    print(predictions)
    return numpy.argmax(predictions)

def PlotImage(image, title):
    imageAsInput = image.reshape(required_image_shape)
    imgplot = plt.imshow(imageAsInput, cmap="binary")
    plt.colorbar(imgplot)
    plt.legend()
    plt.title(title)
    plt.show()


g_command_format = f'python {sys.argv[0]} (model_path) (input_path)'


model_path = "model_mnist_ocr.h5"



input_image_name = "img6.bmp"
outout_image_name = "uart_raw_image_bytes_6.bin"

try:
    image = Image.open(input_image_name)
except:
    FatalError(f'Invalid image: \"{input_image_name}\"')


# Filter pipeline

filterPipeline = FilterPipeline(
    (
        ResizeFilter(required_image_shape), # Resize image to 28x28
        GrayscaleFilter(),                  # Color image to grayscale
        # InvertImageFilter(),                # MNIST dataset is preprocessed to be white ink on black background
        ToNdarrayFilter(),                  # Convert from Image object to ndarray
        # ByteNormalizationFilter(),          # Normalize all bytes by 255
        CastAs(np.int32),
        Offset(-128),
        CastAs(np.int8),
        FlattenFilter(),                     # Flatten image to 1x784
        ExportBytes(outout_image_name)
    )
)

image = filterPipeline.Execute(image)

print(image)

# Load model
model = keras.models.load_model(model_path)
predictedClass = Predict(model, image) 
PlotImage(image, title=f'Prediction = {predictedClass}')