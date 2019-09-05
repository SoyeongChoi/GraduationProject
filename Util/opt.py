from tensorflow.tools import graph_transforms as trans


temps = trans(in_graph = './frozen_model.pb' , out_graph='temp_opt.pb', inputs=['Input'], outputs = ["outputs/concat"],
transforms = )
trans.
tensorflow/tools/graph_transforms/transform_graph \
--in_graph=tensorflow_inception_graph.pb \
--out_graph=optimized_inception_graph.pb \
--inputs='Mul' \
--outputs='softmax' \
--transforms='
  strip_unused_nodes(type=float, shape="1,299,299,3")
  remove_nodes(op=Identity, op=CheckNumerics)
  fold_constants(ignore_errors=true)
  fold_batch_norms
