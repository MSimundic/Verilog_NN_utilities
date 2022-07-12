import numpy as np
import matplotlib.pyplot as plt
# Add plt.show()

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import confusion_matrix
import seaborn as sns

import tensorflow as tf

model_name = 'model_mnist_ocr'

np.random.seed(0)

g_console_title_delimiter = "=" * 16
g_console_subtitle_delimiter = '-' * 16

def printTitle(title : str):
    print(f'{g_console_title_delimiter} {title} {g_console_title_delimiter}')

def printSubtitle(title : str):
    print(f'{g_console_subtitle_delimiter} {title} {g_console_subtitle_delimiter}')

def printNewLine():
    print('\n')

# Data
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'Shape of training dataset: (x, y) = ({x_train.shape}, {y_train.shape})')
print(f'Shape of testing dataset: (x, y) = ({x_test.shape}, {y_test.shape})')

num_classes = 10

# Visualize examples
printTitle("Visualizing Examples")

num_classes = 10

f, ax = plt.subplots(1, num_classes, figsize=(20,20))

for i in range(0, num_classes):
    sample = x_train[y_train == i][0]
    imgplt = ax[i].imshow(sample, cmap='binary')
    f.colorbar(imgplt, ax=ax[i])
    ax[i].set_title(f"Label: {i}", fontsize=16)

plt.show()


printTitle("Examples of classes [outputs]")
printSubtitle("Before categorization")

for i in range(10):
    print(y_train[i])

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

printSubtitle("After categorization")

for i in range(10):
    print(y_train[i])


# Prepare data
# Normalize data
printTitle('Preparing data')
printSubtitle('Normalizing data')

x_train = x_train.astype('int8')
x_test = x_test.astype('int8')

x_train -= 128
x_test -= 128

f, ax = plt.subplots(figsize=(20,20))
sample = x_train[0]
imgplt = ax.imshow(sample, cmap='binary')
f.colorbar(imgplt, ax=ax)
plt.title(f'Prediction: {np.argmax(y_train[0], 0)}')
plt.show()

printSubtitle('Reshaping data')
print(f'Input size before reshaping: (x, y) = 7{x_train.shape}')
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(f'Input size after reshaping: (x, y) = 7{x_train.shape}')


# Create Model
# Fully Connected NN

printTitle('Creating model')

model = Sequential()

# model.add(Dense(units=128, input_shape=(784,), activation='relu', use_bias=False))
model.add(Dense(units=10, input_shape=(784,), activation='softmax', use_bias=False))
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(units=10, activation='softmax'))
# model.add(Dense(units=10))

# model.add(Conv2D(64, (3, 3), input_shape=(28,28,1)))
# model.add(MaxPooling2D((2,2)))
# model.add(Conv2D(128, (3, 3)))
# model.add(MaxPooling2D((2,2)))
# model.add(Flatten())
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

printSubtitle('Model Summary')
model.summary()


# Train
printTitle('Model Training')

batch_size = 512
epochs = 10

# # DEBUG START
# x_train = np.expand_dims(x_train[0], 0) 
# y_train = np.expand_dims(y_train[0], 0) 
# batch_size = 1
# epochs = 1
# # DEBUG END

model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)


# Evaluate
printSubtitle('Evaluating on test datasets')
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

# Confusion matrix
printTitle('Confusion Matrix')
y_true_classes = np.argmax(y_test, axis=1)
y_predicted = model.predict(x_test)
y_predicted_classes = np.argmax(y_predicted, axis=1)
confusion_mtx = confusion_matrix(y_true_classes, y_predicted_classes)

fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt="d", ax=ax, cmap="Reds")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Real Label')
ax.set_title('Confusion Matrix')
plt.show()

# Save model
printTitle('Saving Model')

model.save(model_name + '.h5', overwrite=True, save_format='h5')
print(f'Model saved to \"{model_name}\"')

print(f'Performing full integer quantization...')

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train.astype('float32')).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = (representative_data_gen)
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

model_name = f'{model_name}_quant'
with open(model_name + '.tflite', 'wb') as f:
  f.write(tflite_model_quant)

print(f'Model saved to \"{model_name}\"')