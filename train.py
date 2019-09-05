import tensorflow as tf
from tensorflow import keras
import os 
import numpy as np
import matplotlib.pyplot as plt
import SSDLoss_new as SSDLoss
import json
from keras.callbacks import LambdaCallback
from ssd_utils import BBoxUtility
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy.misc import imread
from Generator import Generator
import pickle
import pandas as pd
import csv
#import keras_tfrecord as ktfr

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3" 



'''
testing for 
class MyCallback:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def testmodel(self, epoch, logs=None):

        epoch
        truex, truey = self.x_train, self.y_train
        predy = train_model.predict(truex)
        print(predy.shape, predy)
        print(truey)
'''
from keras import callbacks
class PlotLearning(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(self.x, self.loss, 'b-', label="train_loss")
        ax1.plot(self.x, self.val_loss, 'r-', label="val_loss")
        ax1.set_title('learning curve')
        ax1.legend(loc=1, prop={'size': 15})

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(self.x, self.acc, 'b-', label="train_acc")
        ax2.plot(self.x, self.val_acc, 'r-', label="val_acc")
        ax2.set_title('accuracy')
        ax2.legend(loc=1, prop={'size': 15})


#print(train_images.shape)
#print(test_images.shape)
def build_keras_model(input_shape, num_classes):
 
    # 
    class_num = num_classes    
    inputs =  tf.keras.layers.Input(shape=input_shape, name="Input")
    # layer 1 : Conv2d + BN 
    # (300,300,3) => (150,150,32)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 2 : DWConv2d + BN 
    # (150,150,32) => (150,150,32)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 3 : PWConv2d + BN 
    # (150,150,32) => (150,150,64)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 4 : DWConv2d + BN 
    # (150,150,64) => (75,75,64)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 5 : PWConv2d + BN 
    # (75,75,64) => (75,75,128)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 6 : DWConv2d + BN 
    # (75,75,128) => (75,75,128)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 7 : PWConv2d + BN 
    # (75,75,128) => (75,75,128)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 8 : DWConv2d + BN 
    # (75,75,128) => (38,38,128)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 9 : PWConv2d + BN 
    # (38,38,128) => (38,38,256)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 10 : DWConv2d + BN 
    # (38,38,256) => (38,38,256)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 11 : PWConv2d + BN 
    # (38,38,256) => (38,38,256)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 12 : DWConv2d + BN 
    # (38,38,256) => (19,19,256)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 13 : PWConv2d + BN 
    # (19,19,256) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # Same 5 times DW + PW : 1
    # layer 14 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 15 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # Same 5 times DW + PW : 2
    # layer 16 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 17 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # Same 5 times DW + PW : 3
    # layer 18 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 19 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # Same 5 times DW + PW : 4
    # layer 20 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 21 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # Same 5 times DW + PW : 5
    # layer 22 : DWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
        
    # layer 23 : PWConv2d + BN 
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)

    
    # feature layer 1 : 
    # (19,19,512) => (19,19, 4 * (class_num | 4)) 
    f1_loc = tf.keras.layers.Conv2D((3 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
    f1_loc = tf.keras.layers.Reshape(( 19 * 19 * 3, (4)), input_shape=(19,19, 3 * (4)))(f1_loc)
        
    f1_conf = tf.keras.layers.Conv2D((3 * (class_num)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
    f1_conf = tf.keras.layers.Reshape(( 19 * 19 * 3, (class_num)), input_shape=(19,19, 3 * (class_num)), name = 'f1_conf')(f1_conf)
       

    # layer 24,25 : DWConv2d + PWConv2d
    # (19,19,512) => (10,10,512) => (10,10,1024)    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
     
    x = tf.keras.layers.Conv2D(1024, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    

    # layer 26,27 : DWConv2d + PWConv2d
    # (10,10,1024) => (10,10,1024) => (10,10,1024)    
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)
    
    x = tf.keras.layers.Conv2D(1024, kernel_size=1, strides=1, activation=None    , padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(fused=False)(x)
    x = tf.keras.layers.ReLU(6.)(x)

    # layer feature 2 : 
    # (10,10,1024) => (10, 10, 6 * (class_num | 4)) 
    f2_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
    f2_loc = tf.keras.layers.Reshape(( 10 * 10 * 6, (4)), input_shape=(10, 10, 6 * (4)))(f2_loc)
   
    f2_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
    f2_conf = tf.keras.layers.Reshape(( 10 * 10 * 6, (class_num)), input_shape=(10, 10, 6 * (class_num)), name = 'f2_conf')(f2_conf)
    
    # layer 28,29 : PWConv2d + Conv2d
    # (10,10,1024) => (10,10,256) => (5,5,512)    
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)
    
    # layer feature 3 : 
    # (5,5,512) => (5, 5, 6 * (class_num | 4)) 
    f3_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
    f3_loc = tf.keras.layers.Reshape(( 5 * 5 * 6, (4)), input_shape=(5, 5, 6 * (4)) )(f3_loc)
    
    f3_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
    f3_conf = tf.keras.layers.Reshape(( 5 * 5 * 6, (class_num)), input_shape=(5, 5, 6 * (class_num)), name = 'f3_conf' )(f3_conf)
    
    # layer 30,31 : PWConv2d + Conv2d
    # (5,5,512) => (5,5,128) => (3,3,256)    
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)
    
    
    # layer feature 4 : 
    # (3,3,256) => (3, 3, 4 * (class_num | 4)) 
    f4_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
    f4_loc = tf.keras.layers.Reshape(( 3 * 3 * 6, (4)),  input_shape=(3, 3, 6 * (4)) )(f4_loc)
    
    f4_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
    f4_conf = tf.keras.layers.Reshape(( 3 * 3 * 6, (class_num)),  input_shape=(3, 3, 6 * (class_num)), name = 'f4_conf' )(f4_conf)
    
    # layer 32,33 : PWConv2d + Conv2d
    # (3,3,256) => (3,3,128) => (2,2,256)    
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)
    
    # layer feature 5 : 
    # (2,2,256) => (2, 2, 4 * (class_num | 4)) 
    f5_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
    f5_loc = tf.keras.layers.Reshape(( 2 * 2 * 6, (4)),  input_shape=(1, 1, 6 * (4)) )(f5_loc)
    
    f5_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
    f5_conf = tf.keras.layers.Reshape(( 2 * 2 * 6, (class_num)),  input_shape=(1, 1, 6 * (class_num)), name = 'f5_conf' )(f5_conf)
    
    # layer 34,35 : PWConv2d + Conv2d
    # (2,2,256) => (2,2,128) => (1,1,256)    
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)
    

    # layer feature 6 : 
    # (1,1,256) => (1, 1, 4 * (class_num | 4)) 
    f6_loc = tf.keras.layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
    f6_loc = tf.keras.layers.Reshape(( 1 * 1 * 6, (4)),  input_shape=(1, 1, 6 * (4)) )(f6_loc)
    
    f6_conf = tf.keras.layers.Conv2D((6 * (class_num)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
    f6_conf = tf.keras.layers.Reshape(( 1 * 1 * 6, (class_num)),  input_shape=(1, 1, 6 * (class_num)) , name = 'f6_conf')(f6_conf)

    z_loc = tf.keras.layers.concatenate([f1_loc, f2_loc, f3_loc, f4_loc, f5_loc, f6_loc], axis=1, name='total_loc')

    z_conf = tf.keras.layers.concatenate([f1_conf, f2_conf, f3_conf, f4_conf, f5_conf, f6_conf], axis=1, name='total_conf')
    z_conf_logistic = tf.keras.layers.Activation('softmax')(z_conf)

    z_concat = tf.keras.layers.concatenate([z_conf_logistic, z_loc], name='outputs')






    
    ### for SSD 5 features contract
        
  
    # return tf.keras.Model(inputs = inputs, outputs = [z_loc, z_conf])

    return tf.keras.Model(inputs = inputs, outputs = z_concat)

# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)


SSDL = SSDLoss.SSDLoss()

keras.backend.set_session(train_sess)
with train_graph.as_default():
    classes = 5
    NUM_CLASSES = classes+1
    input_shape = (300,300,3) 
    train_model = build_keras_model(input_shape, num_classes = NUM_CLASSES)
    
    priors = pickle.load(open('new_ssd_300.pkl', 'rb'))
    #print('@@@@@@',priors)
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    
    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    
        
        
  #  x_train_, y_train_ = ktfr.read_and_decode('train.record', one_hot=True, n_class=nb_classes, is_train=True)
  #  x_test_, y_test_ = ktfr.read_and_decode('test.record', one_hot=True, n_class=nb_classes, is_train=True)
    
   # train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
  #  val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

      
    
    # gt = pickle.load(open('./PASCAL_VOC/VOC2007.p', 'rb'))
    gt = pickle.load(open('./face_data.p', 'rb'))
    
    # path_prefix = '../VOCdevkit/VOC2007/JPEGImages/'
    path_prefix = '../dataset/re_img/'
    
    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

   
    
    
    gen = Generator(gt, bbox_util, 16, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)

                
    train_model.compile(
        optimizer=keras.optimizers.Adam(lr=0.0001),
        # loc, conf loss
        # loss={'total_loc' :SSDL.compute_loss_loc, 'total_conf' : SSDL.compute_loss_conf} ,
        loss =SSDL.compute_loss,
        # loss = 'mse',
        metrics =['accuracy'])
        # metrics=['sparse_categorical_accuracy'])            
    nb_epoch = 100
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=0.000001, verbose=1)
                              
    plot_learning = PlotLearning()
    
    callbacks = [keras.callbacks.ModelCheckpoint('./save_weight/testing_weights_test.{epoch:02d}-{val_loss:.2f}.h5',
                                             verbose=1,
                                             save_weights_only=True), reduce_lr, plot_learning]
                                             
    train_model.load_weights('./save_weight/testing_weights_test.59-8.68.h5')
                                             
                                            
    history = train_model.fit_generator(gen.generate(True), gen.train_batches,
                              nb_epoch, verbose=1,
                              validation_data=gen.generate(False),validation_steps=1,callbacks=callbacks)
    
    # print(train_images.shape, train_labels.shape, train_labels_loc.shape)
  #  train_model.fit(train_images, { 'total_loc': train_labels_loc, 'total_conf' : train_labels_conf} , epochs=1)
    
    # train_model.save_weights('ssd_weight.h5')
    # cb_testmodel = MyCallback(train_images, train_labels)
    # cb_lambda = LambdaCallback(on_epoch_end=cb_testmodel.testmodel)
    #history = train_model.fit(
    
    # load to dict
    # my_dict = json.loads(input) 
    
    
    train_model.save('ssd.h5')


    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, './checkpoints')


with train_graph.as_default():
    print('sample result of original model')
    # print(train_model.predict(test_images[:1]))
# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    NUM_CLASSES = classes+1
    input_shape = (300,300,3) 
    eval_model = build_keras_model(input_shape, num_classes = NUM_CLASSES)
    
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'checkpoints')

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
        # [eval_model.output[0].op.name, eval_model.output[1].op.name]
    )

    with open('./frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

# TODO : tf 버전에 맞게 변경해야함.
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('./frozen_model.pb', input_arrays=['Input'], output_arrays=['outputs/concat'])

# converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('./frozen_model.pb', input_arrays=['Input'], output_arrays=['total_loc/concat', 'total_conf/concat'])

converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats  = { "Input" : (0,255)}
converter.default_ranges_stats = [0,255]


tflite_model = converter.convert()

open("./tflite_model.tflite", "wb").write(tflite_model)




