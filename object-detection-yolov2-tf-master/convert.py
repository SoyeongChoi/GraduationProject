import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
import os

    
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

MODEL_NAME = 'face_model'

# Freeze the graph

input_graph_path = './tmp/'+MODEL_NAME+'.pbtxt'

with tf.Session() as sess:
    
    
    saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
    saver.restore(sess, './tmp/model.ckpt')
    
    checkpoint_path = './tmp/model.ckpt'
    input_saver_def_path = ""
    input_binary = False
    output_node_names = "output"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = './tmp/frozen_'+MODEL_NAME+'.pb'
    
    clear_devices = True
    


    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")
    


output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)


output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def,
    ["input"], # an array of the input node(s)
    ["output"], # an array of output nodes
    tf.float32.as_datatype_enum)




with tf.gfile.FastGFile('./tmp/'+output_optimized_graph_name, "wb") as f:
    f.write(output_graph_def.SerializeToString())



# input shape need!
graph_def_file = './tmp/optimized_'+MODEL_NAME+'.pb'
input_arrays = ["input"]
output_arrays = ["output"]

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays=['input'], input_shapes={'input':[1,416,416,3]} , output_arrays=output_arrays)
# converter.allow_custom_ops = True
# converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
# converter.inference_input_type  = tf.contrib.lite.constants.QUANTIZED_UINT8
# converter.quantized_input_stats  = {'input':(127,127)}
# converter.default_ranges_stats = (0, 255)

tflite_model = converter.convert()

open("converted_model.tflite", "wb").write(tflite_model)
