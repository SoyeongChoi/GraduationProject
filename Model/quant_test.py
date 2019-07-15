import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.reshape(train_images, [-1, 28, 28, 1])
test_images = np.reshape(test_images, [-1, 28, 28, 1])

#test_소영
#Test3
#Test4
open("./model.tflite", "wb").write(tflite_model)

# load TFLite file
interpreter = tf.contrib.lite.Interpreter(model_path='./model.tflite')
# Allocate memory.
interpreter.allocate_tensors()

# get some informations .
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']
    
    return (data/a + b).astype(dtype).reshape(shape)


def dequantize(detail, data):
    a, b = detail['quantization']
    
    return (data - b)*a
    
quantized_input = quantize(input_details[0], test_images[:1])
interpreter.set_tensor(input_details[0]['index'], quantized_input)

interpreter.invoke()

# The results are stored on 'index' of output_details
quantized_output = interpreter.get_tensor(output_details[0]['index'])

print('sample result of quantized model')
print(dequantize(output_details[0], quantized_output))
