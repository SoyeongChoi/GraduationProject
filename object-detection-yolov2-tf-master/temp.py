
import os
from tensorflow.python.platform import gfile
import tensorflow as tf

import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"




GRAPH_PB_PATH = './tmp/frozen_face_model.pb' #path to your .pb file

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    
wts = [n for n in graph_nodes if n.op=='Const']
    
from tensorflow.python.framework import tensor_util

f = open("pb_weights.txt","w+")

for n in wts:
	
    f.write("Name of the node -" + n.name)
    f.write("Value - " )
    f.write(np.array2string(tensor_util.MakeNdarray(n.attr['value'].tensor)))

f.close()
