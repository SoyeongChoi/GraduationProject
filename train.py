import os
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import callbacks
from tensorflow import keras

from data.generator import Generator
from net import loss as SSDLoss
from net.ssd import SSD
from utils.ssd_utils import BBoxUtility

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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


# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

ssd_loss = SSDLoss.SSDLoss()

keras.backend.set_session(train_sess)
with train_graph.as_default():
    classes = 5
    NUM_CLASSES = classes + 1
    input_shape = (300, 300, 3)
    train_model = SSD(input_shape, num_classes=NUM_CLASSES)

    priors = pickle.load(open('new_ssd_300.pkl', 'rb'))

    bbox_util = BBoxUtility(NUM_CLASSES, priors)

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    gt = pickle.load(open('./face_data.p', 'rb'))

    path_prefix = '../dataset/re_img/'

    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

    gen = Generator(gt, bbox_util, 16, path_prefix,
                    train_keys, val_keys,
                    (input_shape[0], input_shape[1]), do_crop=False)

    train_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                        loss=ssd_loss.compute_loss,
                        metrics=['accuracy'])
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
                                        validation_data=gen.generate(False), validation_steps=1, callbacks=callbacks)

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
    NUM_CLASSES = classes + 1
    input_shape = (300, 300, 3)
    eval_model = SSD(input_shape, num_classes=NUM_CLASSES)

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
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('./frozen_model.pb', input_arrays=['Input'],
                                                              output_arrays=['outputs/concat'])

# converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('./frozen_model.pb', input_arrays=['Input'], output_arrays=['total_loc/concat', 'total_conf/concat'])

converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {"Input": (0, 255)}
converter.default_ranges_stats = [0, 255]

tflite_model = converter.convert()

open("./tflite_model.tflite", "wb").write(tflite_model)
