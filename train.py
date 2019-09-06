import os
import pickle

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras

import model.ssd as ssdmodel
import training.loss_function as ssdloss
from generator.generator import Generator as genertor
from util.ssd_utils import BBoxUtility as bboxutility

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

SSDL = ssdloss.SSDLoss()

keras.backend.set_session(train_sess)
with train_graph.as_default():
    classes = ['neural', 'smile', 'anger', 'surprise', 'sad']
    # add 1 for background class
    NUM_CLASSES = len(classes) + 1
    input_shape = (300, 300, 3)

    # make model
    model = ssdmodel.SSD(input_shape, NUM_CLASSES)

    train_model = model.make_model()

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    # load prior_box information
    priors = pickle.load(open('new_ssd_300.pkl', 'rb'))
    bbox_util = bboxutility(NUM_CLASSES, priors)

    # load face data's ground truth information
    gt = pickle.load(open('./face_data.p', 'rb'))

    path_prefix = '../dataset/re_img/'

    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

    # generate dataset using 'gt', 'prior_box', 'BBoxUtility'
    gen = genertor(gt, bbox_util, 16, path_prefix,
                   train_keys, val_keys,
                   (input_shape[0], input_shape[1]), do_crop=False)

    train_model.compile(
        optimizer=keras.optimizers.Adam(lr=0.0001),
        loss=SSDL.compute_loss,
        metrics=['accuracy'])

    nb_epoch = 100

    # callbacks
    model_chekpoint = ModelCheckpoint('./save_weight/testing_weights_test.{epoch:02d}-{val_loss:.2f}.h5',
                                      verbose=1,
                                      save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.000001, verbose=1)

    csv_logger = CSVLogger(filename='Face_new_box_training_log.csv', separator=',', append=True)

    callbacks = [model_chekpoint, reduce_lr, csv_logger]

    # load weight                                         
    # train_model.load_weights('./save_weight/testing_weights_test.59-8.68.h5')

    print(train_model.summary())
    # start training                                        
    history = train_model.fit_generator(gen.generate(True), gen.train_batches,
                                        nb_epoch, verbose=1,
                                        validation_data=gen.generate(False), validation_steps=1, callbacks=callbacks)

    # save weight with architecture
    train_model.save('./save_weight/ssd.h5')

    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, './checkpoints')

# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    NUM_CLASSES = len(classes) + 1
    input_shape = (300, 300, 3)

    eval_model = model.make_model()

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

# TODO : modify code <tf version>
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('./frozen_model.pb', input_arrays=['Input'],
                                                              output_arrays=['outputs/concat'])

# converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('./frozen_model.pb', input_arrays=['Input'], output_arrays=['total_loc/concat', 'total_conf/concat'])

converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {"Input": (0, 255)}
converter.default_ranges_stats = [0, 255]

tflite_model = converter.convert()

open("./tflite_model.tflite", "wb").write(tflite_model)
