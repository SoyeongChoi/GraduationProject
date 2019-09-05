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

from imageio import imread
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
    classes = ['neural','smile','anger', 'surprise' , 'sad' ]
    
    NUM_CLASSES = len(classes)+1
    input_shape = (300,300,3) 
    train_model = build_keras_model(input_shape, num_classes = NUM_CLASSES)
    
    priors = pickle.load(open('new_ssd_300.pkl', 'rb'))
    bbox_util = BBoxUtility(NUM_CLASSES, priors)
    print(priors)
    print(priors.shape)
    # tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    
    # train_sess.run(tf.global_variables_initializer())

    
        
        
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
    

    
    gen = Generator(gt, bbox_util, 16, '../dataset/img/',
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)
                
    train_model.compile(
        optimizer='adam',
        # loc, conf loss
        # loss={'total_loc' :SSDL.compute_loss_loc, 'total_conf' : SSDL.compute_loss_conf} ,
        loss =SSDL.compute_loss,
        # loss = 'mse',
        metrics =['accuracy'])
        # metrics=['sparse_categorical_accuracy'])            
    # nb_epoch = 5
    # history = train_model.fit_generator(gen.generate(True), gen.train_batches,
    #                           nb_epoch, verbose=1,
    #                           validation_data=gen.generate(False),validation_steps=1)
    # print(train_images.shape, train_labels.shape, train_labels_loc.shape)
  #  train_model.fit(train_images, { 'total_loc': train_labels_loc, 'total_conf' : train_labels_conf} , epochs=1)
    
    # train_model.save_weights('ssd_weight.h5')
    # cb_testmodel = MyCallback(train_images, train_labels)
    # cb_lambda = LambdaCallback(on_epoch_end=cb_testmodel.testmodel)
    #history = train_model.fit(
    
    # load to dict
    # my_dict = json.loads(input) 
    train_model.load_weights('./save_weight/testing_weights_test.59-8.68.h5')
    
    
    path_prefix = '../dataset/re_img/'
    inputs = []
    images = []
    
    temp_list = os.listdir(path_prefix)
    
    for x in range(20,50):
      img_path = path_prefix + temp_list[x]
      
      img = image.load_img(img_path, target_size=(300, 300))
      img = image.img_to_array(img)
      images.append(imread(img_path))
      inputs.append(img.copy())
      
    inputs = preprocess_input(np.array(inputs))


    
    preds = train_model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out2(preds)


    k=0
    for i, img in enumerate(images):
        # Parse the outputs.
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]
        
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]
    
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
    
        colors = plt.cm.hsv(np.linspace(0, 1, 6)).tolist()
    
        plt.imshow(img / 255.)
        currentAxis = plt.gca()
    
        
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        
        # plt.show()
        plt.savefig('./result/' + str(k) +'.jpg')
        plt.clf()
        k= k + 1



