import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from tensorflow import keras

import model.ssd as ssdmodel
import training.loss_function as ssdloss
from generator.generator import Generator as genertor
from util.ssd_utils import BBoxUtility as bboxutility

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

SSDL = ssdloss.SSDLoss()

classes = ['neural', 'smile', 'anger', 'surprise', 'sad']

# add 1 for background class
NUM_CLASSES = len(classes) + 1
input_shape = (300, 300, 3)

# make model
model = ssdmodel.SSD(input_shape, NUM_CLASSES)

train_model = model.make_model()

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
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=SSDL.compute_loss,
    metrics=['accuracy'])

# load weight
train_model.load_weights('./save_weight/testing_weights_test.59-8.68.h5')

# start testing

inputs = []
images = []

temp_list = os.listdir(path_prefix)

for x in range(20, 50):
    img_path = path_prefix + temp_list[x]

    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())

inputs = preprocess_input(np.array(inputs))

preds = train_model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out2(preds)

k = 0
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
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    # plt.show()
    plt.savefig('./result/' + str(k) + '.jpg')
    plt.clf()
    k = k + 1
