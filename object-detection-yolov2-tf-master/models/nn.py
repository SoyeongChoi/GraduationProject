import time
from abc import abstractmethod, ABCMeta
import tensorflow as tf
import numpy as np
from models.layers import conv_layer, max_pool, fc_layer, batchNormalization, depthwise_conv2d


class DetectNet(metaclass=ABCMeta):
    """Base class for Convolutional Neural Networks for detection."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        model initializer
        :param input_shape: tuple, shape (H, W, C)
        :param num_classes: int, total number of classes
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape,  name = 'input')
        self.is_train = tf.placeholder(tf.bool)
        self.num_classes = num_classes
        self.d = self._build_model(**kwargs)
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)


    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        build loss function for the model training.
        This should be implemented.
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
                -batch_size: int, batch size for each iteration.
        :return _y_pred: np.ndarray, shape: shape of self.pred
        """

        batch_size = kwargs.pop('batch_size', 16)

        num_classes = self.num_classes
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size
        flag = int(bool(pred_size % batch_size))
        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps + flag):
            if i == num_steps and flag:
                _batch_size = pred_size - num_steps * batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False)

            # Compute predictions
            # (N, grid_h, grid_w, 5 + num_classes)
            y_pred = sess.run(self.pred_y, feed_dict={
                              self.X: X, self.is_train: False})

            _y_pred.append(y_pred)

        if verbose:
            print('Total prediction time(sec): {}'.format(
                time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)
        return _y_pred


class YOLO(DetectNet):
    """YOLO class"""

    def __init__(self, input_shape, num_classes, anchors, **kwargs):

        self.grid_size = grid_size = [x // 32 for x in input_shape[:2]]
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.y = tf.placeholder(tf.float32, [None] +
                                [self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + num_classes], name = 'input2')
        super(YOLO, self).__init__(input_shape, num_classes, **kwargs)

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building YOLO.
                -image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """

        d = dict()
        x_mean = kwargs.pop('image_mean', 0.0)

        # input
        X_input = self.X - x_mean
        is_train = self.is_train

        #conv1 - relu61 - pool1
        with tf.variable_scope('layer1'):
            d['conv1'] = conv_layer(X_input, 3, 2, 32, padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:0')
                                    
            d['relu61'] = tf.nn.relu6(d['conv1'],   name='relu61_output')
            
            
        # (416, 416, 3) --> (208, 208, 32)
        print('layer1.shape', d['relu61'].get_shape().as_list())

        #dwconv2 - relu62 - pool2
        with tf.variable_scope('layer2'):
            d['conv2'] = depthwise_conv2d(d['relu61'], 32,32, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:1')
        
            d['relu62'] = tf.nn.relu6(d['conv2'],   name='relu62_output')
        # (208, 208, 32) --> (208, 208, 32)
        print('layer2.shape', d['relu62'].get_shape().as_list())

        #pwconv3 - relu63
        with tf.variable_scope('layer3'):
            d['conv3'] = conv_layer(d['relu62'], 1, 2, 64,
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:3')
            d['relu63'] = tf.nn.relu6(d['conv3'],   name='relu63_output')
            
        # (208, 208, 32) --> (104, 104, 64)
        print('layer3.shape', d['relu63'].get_shape().as_list())

        #dwconv4 - relu64
        with tf.variable_scope('layer4'):
            d['conv4'] = depthwise_conv2d(d['relu63'], 64,64, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:4')
            d['relu64'] = tf.nn.relu6(d['conv4'],   name='relu64_output')
        # (104, 104, 64) --> (104, 104, 64)
        print('layer4.shape', d['relu64'].get_shape().as_list())

        #pwconv5  - relu65 - pool5
        with tf.variable_scope('layer5'):
            d['conv5'] = conv_layer(d['relu64'], 1, 2, 128,
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:5')
            d['relu65'] = tf.nn.relu6(d['conv5'],   name='relu65_output')
        
        # (104, 104, 64) --> (52, 52, 128)
        print('layer5.shape', d['relu65'].get_shape().as_list())

        #dwconv6  - relu66
        with tf.variable_scope('layer6'):
            d['conv6'] = depthwise_conv2d(d['relu65'],128,128, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:6')
            d['relu66'] = tf.nn.relu6(d['conv6'],   name='relu66_output')
        # (52, 52, 128) --> (52, 52, 128)
        print('layer6.shape', d['relu66'].get_shape().as_list())

        #pwconv7  - relu67
        with tf.variable_scope('layer7'):
            d['conv7'] = conv_layer(d['relu66'], 1, 1, 256, use_bias=False,
                                    padding='SAME', weights_stddev=0.01, biases_value=0.01, name='Conv2D:7')
            d['relu67'] = tf.nn.relu6(d['conv7'],   name='relu67_output')
            
          
        # (52, 52, 128) --> (52, 52, 256)
        print('layer7.shape', d['relu67'].get_shape().as_list())

        #dwconv8 - relu68 - pool8
        with tf.variable_scope('layer8'):
            d['conv8'] = depthwise_conv2d(d['relu67'], 256,256, strides=[1,2,2,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:8')
            d['relu68'] = tf.nn.relu6(d['conv8'],   name='relu68_output')
        # (52, 52, 256) --> (26, 26, 256)
        print('layer8.shape', d['conv8'].get_shape().as_list())

        #pwconv9  - relu69
        with tf.variable_scope('layer9'):
            d['conv9'] = conv_layer(d['conv8'], 1, 1, 512,
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:9')
            d['relu69'] = tf.nn.relu6(d['conv9'],   name='relu69_output')
        # (26, 26, 256) --> (26, 26, 512)
        print('layer9.shape', d['relu69'].get_shape().as_list())

        #dwconv10 - relu610
        with tf.variable_scope('layer10'):
            d['conv10'] =  depthwise_conv2d(d['relu69'], 512,512, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:10')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer10.shape', d['conv10'].get_shape().as_list())

        #pwconv11 -  relu611
        with tf.variable_scope('layer11'):
            d['conv11'] = conv_layer(d['conv10'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:11')
            d['relu611'] = tf.nn.relu6(d['conv11'],   name='relu611_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer11.shape', d['relu611'].get_shape().as_list())

        #dwconv12 -  relu612
        with tf.variable_scope('layer12'):
            d['conv12'] =  depthwise_conv2d(d['relu611'], 512,512, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:12')
            d['relu612'] = tf.nn.relu6(d['conv12'],   name='relu612_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer12.shape', d['relu612'].get_shape().as_list())

        #pwconv13 - relu613 
        with tf.variable_scope('layer13'):
            d['conv13'] = conv_layer(d['relu612'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:13')
            d['relu613'] = tf.nn.relu6(d['conv13'],   name='relu613_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer13.shape', d['relu613'].get_shape().as_list())

        #dwconv14 - relu614
        with tf.variable_scope('layer14'):
            d['conv14'] = depthwise_conv2d(d['relu613'], 512,512, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:14')
            d['relu614'] = tf.nn.relu6(d['conv14'],   name='relu614_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer14.shape', d['relu614'].get_shape().as_list())

        #pwconv15 - batch_norm15 - relu615
        with tf.variable_scope('layer15'):
            d['conv15'] = conv_layer(d['relu614'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:15')
            d['relu615'] = tf.nn.relu6(d['conv15'],   name='relu615_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer15.shape', d['relu615'].get_shape().as_list())

        #dwconv16 - batch_norm16 - relu616
        with tf.variable_scope('layer16'):
            d['conv16'] = depthwise_conv2d(d['relu615'], 512,512, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:16')
            d['relu616'] = tf.nn.relu6(d['conv16'],   name='relu616_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer16.shape', d['relu616'].get_shape().as_list())

        #pwconv17 - relu617
        with tf.variable_scope('layer17'):
            d['conv17'] = conv_layer(d['relu616'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:17')
            d['relu617'] = tf.nn.relu6(d['conv17'],   name='relu617_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer17.shape', d['relu617'].get_shape().as_list())

        #dwconv18 - relu618
        with tf.variable_scope('layer18'):
            d['conv18'] = depthwise_conv2d(d['relu617'], 512,512, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:18')
            d['relu618'] = tf.nn.relu6(d['conv18'],   name='relu618_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer18.shape', d['relu618'].get_shape().as_list())

        #pwconv19 - relu619
        with tf.variable_scope('layer19'):
            d['conv19'] = conv_layer(d['relu618'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:19')
            d['relu619'] = tf.nn.relu6(d['conv19'],   name='relu619_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer19.shape', d['relu619'].get_shape().as_list())

        #dwconv20 - relu620
        with tf.variable_scope('layer20'):
            d['conv20'] = depthwise_conv2d(d['relu619'], 512,512, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:20')
            d['relu620'] = tf.nn.relu6(d['conv20'],   name='relu620_output')
        # (26, 26, 512) --> (26, 26, 512)
        print('layer20.shape', d['relu620'].get_shape().as_list())

        #pwconv21 -  relu621
        with tf.variable_scope('layer21'):
            d['conv21'] = conv_layer(d['relu620'], 1, 2, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:21')
            d['relu621'] = tf.nn.relu6(d['conv21'],   name='relu621_output')
            
        # (26, 26, 512) --> (13, 13, 1024)       
        print('layer21.shape', d['relu621'].get_shape().as_list())

        #dwconv22 - relu622
        with tf.variable_scope('layer22'):
            d['conv22'] = depthwise_conv2d(d['relu621'], 1024,1024, strides=[1,1,1,1],
                                    padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:22')
            d['relu622'] = tf.nn.relu6(d['conv22'],   name='relu622_output')
        # (13, 13, 1024) --> (13, 13, 1024)
        print('layer22.shape', d['relu622'].get_shape().as_list())
        
        #pwconv23 -  relu622
        with tf.variable_scope('layer23'):
            d['conv23'] = conv_layer(d['relu622'], 1, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01, name='Conv2D:23')
            d['relu623'] = tf.nn.relu6(d['conv23'],   name='relu623_output')
        # (13, 13, 1024) --> (13, 13, 1024)
        print('layer23.shape', d['relu623'].get_shape().as_list())
        

        output_channel = self.num_anchors * (5 + self.num_classes)
        d['logit'] = conv_layer(d['relu623'], 1, 1, output_channel,
                                padding='SAME', use_bias=True, weights_stddev=0.01, biases_value=0.1)
                                
        d['output'] = tf.identity(d['logit'], name = 'output')
        d['pred'] = tf.reshape(
            d['output'], (-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + self.num_classes), name="reshape_output")
            
        
        print('pred.shape', d['pred'].get_shape().as_list())
        # (13, 13, 1024) --> (13, 13, num_anchors , (5 + num_classes))

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments
                - loss_weights: list, [xy, wh, resp_confidence, no_resp_confidence, class_probs]
        :return tf.Tensor.
        """

        loss_weights = kwargs.pop('loss_weights', [5, 5, 5, 0.5, 1.0])
        # DEBUG
        # loss_weights = kwargs.pop('loss_weights', [1.0, 1.0, 1.0, 1.0, 1.0])
        grid_h, grid_w = self.grid_size
        num_classes = self.num_classes
        anchors = self.anchors
        grid_wh = np.reshape([grid_w, grid_h], [
                             1, 1, 1, 1, 2]).astype(np.float32)
        cxcy = np.transpose([np.tile(np.arange(grid_w), grid_h),
                             np.repeat(np.arange(grid_h), grid_w)])
        cxcy = np.reshape(cxcy, (1, grid_h, grid_w, 1, 2))

        txty, twth = self.pred[..., 0:2], self.pred[..., 2:4]
        confidence = tf.sigmoid(self.pred[..., 4:5])
        class_probs = tf.nn.softmax(
            self.pred[..., 5:], axis=-1) if num_classes > 1 else tf.sigmoid(self.pred[..., 5:])
        bxby = tf.sigmoid(txty) + cxcy
        pwph = np.reshape(anchors, (1, 1, 1, self.num_anchors, 2)) / 32
        bwbh = tf.exp(twth) * pwph

        # calculating for prediction
        nxny, nwnh = bxby / grid_wh, bwbh / grid_wh
        nx1ny1, nx2ny2 = nxny - 0.5 * nwnh, nxny + 0.5 * nwnh
        self.pred_y = tf.concat(
            (nx1ny1, nx2ny2, confidence, class_probs), axis=-1)

        # calculating IoU for metric
        num_objects = tf.reduce_sum(self.y[..., 4:5], axis=[1, 2, 3, 4])
        max_nx1ny1 = tf.maximum(self.y[..., 0:2], nx1ny1)
        min_nx2ny2 = tf.minimum(self.y[..., 2:4], nx2ny2)
        intersect_wh = tf.maximum(min_nx2ny2 - max_nx1ny1, 0.0)
        intersect_area = tf.reduce_prod(intersect_wh, axis=-1)
        intersect_area = tf.where(
            tf.equal(intersect_area, 0.0), tf.zeros_like(intersect_area), intersect_area)
        gt_box_area = tf.reduce_prod(
            self.y[..., 2:4] - self.y[..., 0:2], axis=-1)
        box_area = tf.reduce_prod(nx2ny2 - nx1ny1, axis=-1)
        iou = tf.truediv(
            intersect_area, (gt_box_area + box_area - intersect_area))
        sum_iou = tf.reduce_sum(iou, axis=[1, 2, 3])
        self.iou = tf.truediv(sum_iou, num_objects)

        gt_bxby = 0.5 * (self.y[..., 0:2] + self.y[..., 2:4]) * grid_wh
        gt_bwbh = (self.y[..., 2:4] - self.y[..., 0:2]) * grid_wh

        resp_mask = self.y[..., 4:5]
        no_resp_mask = 1.0 - resp_mask
        gt_confidence = resp_mask * tf.expand_dims(iou, axis=-1)
        gt_class_probs = self.y[..., 5:]

        loss_bxby = loss_weights[0] * resp_mask * \
            tf.square(gt_bxby - bxby)
        loss_bwbh = loss_weights[1] * resp_mask * \
            tf.square(tf.sqrt(gt_bwbh) - tf.sqrt(bwbh))
        loss_resp_conf = loss_weights[2] * resp_mask * \
            tf.square(gt_confidence - confidence)
        loss_no_resp_conf = loss_weights[3] * no_resp_mask * \
            tf.square(gt_confidence - confidence)
        loss_class_probs = loss_weights[4] * resp_mask * \
            tf.square(gt_class_probs - class_probs)

        merged_loss = tf.concat((
                                loss_bxby,
                                loss_bwbh,
                                loss_resp_conf,
                                loss_no_resp_conf,
                                loss_class_probs
                                ),
                                axis=-1)
        #self.merged_loss = merged_loss
        total_loss = tf.reduce_sum(merged_loss, axis=-1)
        total_loss = tf.reduce_mean(total_loss)
        return total_loss

    # def interpret_output(self, sess, images, **kwargs):
    #     """
    #     Interpret outputs to decode bounding box from y_pred.
    #     :param sess: tf.Session
    #     :param kwargs: dict, extra arguments for prediction.
    #             -batch_size: int, batch size for each iteraction.
    #     :param images: np.ndarray, shape (N, H, W, C)
    #     :return bbox_pred: np.ndarray, shape (N, grid_h*grid_w*num_anchors, 5 + num_classes)
    #     """
    #     batch_size = kwargs.pop('batch_size', 32)
    #     is_batch = len(images.shape) == 4
    #     if not is_batch:
    #         images = np.expand_dims(images, 0)
    #     pred_size = images.shape[0]
    #     num_steps = pred_size // batch_size

    #     bboxes = []
    #     for i in range(num_steps + 1):
    #         if i == num_steps:
    #             image = images[i * batch_size:]
    #         else:
    #             image = images[i * batch_size:(i + 1) * batch_size]
    #         bbox = sess.run(self.pred_y, feed_dict={
    #                         self.X: image, self.is_train: False})
    #         bbox = np.reshape(bbox, (bbox.shape[0], -1, bbox.shape[-1]))
    #         bboxes.append(bbox)
    #     bboxes = np.concatenate(bboxes, axis=0)

    #     if is_batch:
    #         return bboxes
    #     else:
    #         return bboxes[0]
