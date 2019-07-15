import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

trainI = np.zeros((60000, 300, 300))
testI = np.zeros((10000, 300, 300))



# pad for testing
trainI[:,135:163, 135:163] = train_images
testI[:,135:163, 135:163] = test_images

# print(train_images.shape)
# print(test_images.shape)

train_images = np.reshape(trainI, [-1, 300, 300, 1])
test_images = np.reshape(testI, [-1, 300, 300, 1])

print(train_images.shape)
print(test_images.shape)
def build_keras_model():
 
    # 
    class_num = 5    
    inputs = keras.layers.Input(shape = (300,300,1))
    
    # layer 1 : Conv2d + BN 
    # (300,300,3) => (150,150,32)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False, input_shape=(300, 300, 3))(inputs)
    # x = keras.layers.BatchNormalization(fused=False)(x)(x)
        
    # layer 2 : DWConv2d + BN 
    # (150,150,32) => (150,150,32)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 3 : PWConv2d + BN 
    # (150,150,32) => (150,150,64)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 4 : DWConv2d + BN 
    # (150,150,64) => (75,75,64)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 5 : PWConv2d + BN 
    # (75,75,64) => (75,75,128)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 6 : DWConv2d + BN 
    # (75,75,128) => (75,75,128)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 7 : PWConv2d + BN 
    # (75,75,128) => (75,75,128)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 8 : DWConv2d + BN 
    # (75,75,128) => (38,38,128)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 9 : PWConv2d + BN 
    # (38,38,128) => (38,38,256)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 10 : DWConv2d + BN 
    # (38,38,256) => (38,38,256)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 11 : PWConv2d + BN 
    # (38,38,256) => (38,38,256)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 12 : DWConv2d + BN 
    # (38,38,256) => (19,19,256)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 13 : PWConv2d + BN 
    # (19,19,256) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # Same 5 times DW + PW : 1
    # layer 14 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 15 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # Same 5 times DW + PW : 2
    # layer 16 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 17 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # Same 5 times DW + PW : 3
    # layer 18 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 19 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # Same 5 times DW + PW : 4
    # layer 20 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 21 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # Same 5 times DW + PW : 5
    # layer 22 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
        
    # layer 23 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
    
    # Same 5 times DW + PW : 5
    # layer 22 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
    

    # layer 23 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)
    
    
    # feature layer 1 : 
    # (19,19,512) => (19,19, 4 * (class_num + 4)) 
    f1 = tf.keras.layers.Conv2D((4 * (class_num + 4)), kernel_size=3, activation="relu", padding="same", use_bias=False)(x)
    f1 = tf.keras.layers.Reshape(( 19 * 19 * 4, (class_num + 4)), input_shape=(19,19, 4 * (class_num + 4)))(f1)
        
       
    # layer 24,25 : Conv2d + PWConv2d
    # (19,19,512) => (10,10,1024) => (10,10,1024)    
    x = tf.keras.layers.Conv2D(1024, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    
    
    # layer feature 2 : 
    # (10,10,1024) => (10, 10, 6 * (class_num + 4)) 
    f2 = tf.keras.layers.Conv2D((6 * (class_num + 4)), kernel_size=3, activation="relu", padding="same", use_bias=False)(x)
    f2 = tf.keras.layers.Reshape(( 10 * 10 * 6, (class_num + 4)), input_shape=(10, 10, 6 * (class_num + 4)))(f2)
   
    # layer 26,27 : PWConv2d + Conv2d
    # (10,10,1024) => (10,10,256) => (5,5,512)    
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    
    # layer feature 3 : 
    # (5,5,512) => (5, 5, 6 * (class_num + 4)) 
    f3 = tf.keras.layers.Conv2D((6 * (class_num + 4)), kernel_size=3, activation="relu", padding="same", use_bias=False)(x)
    f3 = tf.keras.layers.Reshape(( 5 * 5 * 6, (class_num + 4)), input_shape=(5, 5, 6 * (class_num + 4)) )(f3)
    
    # layer 28,29 : PWConv2d + Conv2d
    # (5,5,512) => (5,5,128) => (3,3,256)    
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    
    
    # layer feature 4 : 
    # (3,3,256) => (3, 3, 4 * (class_num + 4)) 
    f4 = tf.keras.layers.Conv2D((4 * (class_num + 4)), kernel_size=3, activation="relu", padding="same", use_bias=False)(x)
    f4 = tf.keras.layers.Reshape(( 3 * 3 * 4, (class_num + 4)),  input_shape=(3, 3, 4 * (class_num + 4)) )(f4)
    
    # layer 30,31 : PWConv2d + Conv2d
    # (3,3,256) => (3,3,128) => (1,1,256)    
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    
    # layer feature 5 : 
    # (3,3,256) => (1, 1, 4 * (class_num + 4)) 
    f5 = tf.keras.layers.Conv2D((4 * (class_num + 4)), kernel_size=3, activation="relu", use_bias=False)(x)
    f5 = tf.keras.layers.Reshape(( 1 * 1 * 4, (class_num + 4)),  input_shape=(1, 1, 4 * (class_num + 4)) )(f5)
    ## 
    
    z = tf.keras.layers.Concatenate(axis= -1)([f1, f2, f3, f4, f5])
    
    
    ### for SSD 5 features contract
        
  
    return tf.keras.Model(inputs = inputs, outputs = z)


# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    train_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(train_images.shape, train_labels.shape)
    train_model.fit(train_images, train_labels, epochs=5)

    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, './checkpoints')


with train_graph.as_default():
    print('sample result of original model')
    print(train_model.predict(test_images[:1]))

# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    eval_model = build_keras_model()
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'checkpoints')

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )

    with open('./frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('./frozen_model.pb', input_arrays=['conv2d_input'], output_arrays=['dense/Softmax'])

converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats  = { "conv2d_input" : (0,255)}
converter.default_ranges_stats = [0,255]


tflite_model = converter.convert()
