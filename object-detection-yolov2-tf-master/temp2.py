import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"



sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
saver.restore(sess, './tmp/model.ckpt')
# get the graph
# g = tf.get_default_graph()
# w1 = g.get_tensor_by_name('input:0')


f = open("meta_weights.txt","w+")

for var in tf.all_variables():
    f.write(var.name)
    f.write(np.array2string(sess.run(var)))
f.close()

