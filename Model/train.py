import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2
import keras.utils
import PriorBox


# from adabound import AdaBound

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"


class SSDLoss:
    '''
    The SSD loss, see https://arxiv.org/abs/1512.02325.
    '''

    def __init__(self,
                 neg_pos_ratio=3,
                 n_neg_min=0,
                 alpha=1.0):
        '''
        Arguments:
            neg_pos_ratio (int, optional): The maximum ratio of negative (i.e. background)
                to positive ground truth boxes to include in the loss computation.
                There are no actual background ground truth boxes of course, but `y_true`
                contains anchor boxes labeled with the background class. Since
                the number of background boxes in `y_true` will usually exceed
                the number of positive boxes by far, it is necessary to balance
                their influence on the loss. Defaults to 3 following the paper.
            n_neg_min (int, optional): The minimum number of negative ground truth boxes to
                enter the loss computation *per batch*. This argument can be used to make
                sure that the model learns from a minimum number of negatives in batches
                in which there are very few, or even none at all, positive ground truth
                boxes. It defaults to 0 and if used, it should be set to a value that
                stands in reasonable proportion to the batch size used for training.
            alpha (float, optional): A factor to weight the localization loss in the
                computation of the total loss. Defaults to 1.0 following the paper.
        '''
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = n_neg_min
        self.alpha = alpha

    def smooth_L1_loss(self, y_true, y_pred):
        '''
        Compute smooth L1 loss, see references.
        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape `(batch_size, #boxes, 4)` and
                contains the ground truth bounding box coordinates, where the last dimension
                contains `(xmin, xmax, ymin, ymax)`.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box coordinates.
        Returns:
            The smooth L1 loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        References:
            https://arxiv.org/abs/1504.08083
        '''
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)

    def log_loss(self, y_true, y_pred):
        '''
        Compute the softmax log loss.
        Arguments:
            y_true (nD tensor): A TensorFlow tensor of any shape containing the ground truth data.
                In this context, the expected tensor has shape (batch_size, #boxes, #classes)
                and contains the ground truth bounding box categories.
            y_pred (nD tensor): A TensorFlow tensor of identical structure to `y_true` containing
                the predicted data, in this context the predicted bounding box categories.
        Returns:
            The softmax log loss, a nD-1 Tensorflow tensor. In this context a 2D tensor
            of shape (batch, n_boxes_total).
        '''
        # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
        y_pred = tf.maximum(y_pred, 1e-15)
        # Compute the log loss
        log_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        return log_loss

    def compute_loss(self, y_true, y_pred):
        '''
        Compute the loss of the SSD model prediction against the ground truth.
        Arguments:
            y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
                where `#boxes` is the total number of boxes that the model predicts
                per image. Be careful to make sure that the index of each given
                box in `y_true` is the same as the index for the corresponding
                box in `y_pred`. The last axis must have length `#classes + 12` and contain
                `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
                in this order, including the background class. The last eight entries of the
                last axis are not used by this function and therefore their contents are
                irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
                where the last four entries of the last axis contain the anchor box
                coordinates, which are needed during inference. Important: Boxes that
                you want the cost function to ignore need to have a one-hot
                class vector of all zeros.
            y_pred (Keras tensor): The model prediction. The shape is identical
                to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
                The last axis must contain entries in the format
                `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.
        Returns:
            A scalar, the total multitask loss for classification and localization.
        '''
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)

        batch_size = tf.shape(y_pred)[0]  # Output dtype: tf.int32
        n_boxes = tf.shape(y_pred)[
            1]  # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell.

        # 1: Compute the losses for class and box predictions for every box.

        classification_loss = tf.to_float(
            self.log_loss(y_true[:, :, :-12], y_pred[:, :, :-12]))  # Output shape: (batch_size, n_boxes)
        localization_loss = tf.to_float(
            self.smooth_L1_loss(y_true[:, :, -12:-8], y_pred[:, :, -12:-8]))  # Output shape: (batch_size, n_boxes)

        # 2: Compute the classification losses for the positive and negative targets.

        # Create masks for the positive and negative ground truth classes.
        negatives = y_true[:, :, 0]  # Tensor of shape (batch_size, n_boxes)
        positives = tf.to_float(tf.reduce_max(y_true[:, :, 1:-12], axis=-1))  # Tensor of shape (batch_size, n_boxes)

        # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch.
        n_positive = tf.reduce_sum(positives)

        # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
        # (Keras loss functions must output one scalar loss value PER batch item, rather than just
        # one scalar for the entire batch, that's why we're not summing across all axes).
        pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

        # Compute the classification loss for the negative default boxes (if there are any).

        # First, compute the classification loss for all negative boxes.
        neg_class_loss_all = classification_loss * negatives  # Tensor of shape (batch_size, n_boxes)
        n_neg_losses = tf.count_nonzero(neg_class_loss_all,
                                        dtype=tf.int32)  # The number of non-zero loss entries in `neg_class_loss_all`
        # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
        # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
        # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
        # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
        # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
        # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
        # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
        # is at most the number of negative boxes for which there is a positive classification loss.

        # Compute the number of negative examples we want to account for in the loss.
        # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller).
        n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.to_int32(n_positive), self.n_neg_min),
                                     n_neg_losses)

        # In the unlikely case when either (1) there are no negative ground truth boxes at all
        # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`.
        def f1():
            return tf.zeros([batch_size])

        # Otherwise compute the negative loss.
        def f2():
            # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
            # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
            # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

            # To do this, we reshape `neg_class_loss_all` to 1D...
            neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
            # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
            values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                          k=n_negative_keep,
                                          sorted=False)  # We don't need them sorted.
            # ...and with these indices we'll create a mask...
            negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                           updates=tf.ones_like(indices, dtype=tf.int32),
                                           shape=tf.shape(
                                               neg_class_loss_all_1D))  # Tensor of shape (batch_size * n_boxes,)
            negatives_keep = tf.to_float(
                tf.reshape(negatives_keep, [batch_size, n_boxes]))  # Tensor of shape (batch_size, n_boxes)
            # ...and use it to keep only those boxes and mask all other classification losses
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep,
                                           axis=-1)  # Tensor of shape (batch_size,)
            return neg_class_loss

        neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

        class_loss = pos_class_loss + neg_class_loss  # Tensor of shape (batch_size,)

        # 3: Compute the localization loss for the positive targets.
        #    We don't compute a localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to).

        loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)  # Tensor of shape (batch_size,)

        # 4: Compute the total loss.

        total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive)  # In case `n_positive == 0`
        # Keras has the annoying habit of dividing the loss by the batch size, which sucks in our case
        # because the relevant criterion to average our loss over is the number of positive boxes in the batch
        # (by which we're dividing in the line above), not the batch size. So in order to revert Keras' averaging
        # over the batch size, we'll have to multiply by it.
        total_loss = total_loss * tf.to_float(batch_size)

        return total_loss


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

trainI = np.zeros((5000, 300, 300))
testI = np.zeros((5000, 300, 300))

train_labels = train_labels[0:5000]

# pad for testing
trainI[:, 135:163, 135:163] = train_images[0:5000, :, :]
testI[:, 135:163, 135:163] = test_images[0:5000, :, :]

# print(train_images.shape)
# print(test_images.shape)

train_images = np.reshape(trainI, [5000, 300, 300, 1])
test_images = np.reshape(testI, [5000, 300, 300, 1])

print(train_images.shape)
print(test_images.shape)


def build_keras_model():
    predict_layers_num = 5
    l2_reg = 0.0005
    #
    class_num = 5
    inputs = tf.keras.layers.Input(shape=(300, 300, 1), name="Input")
    # layer 1 : Conv2d + BN
    # (300,300,3) => (150,150,32)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(inputs)
    # x = keras.layers.BatchNormalization(fused=False)(x)(x)

    # layer 2 : DWConv2d + BN
    # (150,150,32) => (150,150,32)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 3 : PWConv2d + BN
    # (150,150,32) => (150,150,64)
    x = tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 4 : DWConv2d + BN
    # (150,150,64) => (75,75,64)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 5 : PWConv2d + BN
    # (75,75,64) => (75,75,128)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 6 : DWConv2d + BN
    # (75,75,128) => (75,75,128)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 7 : PWConv2d + BN
    # (75,75,128) => (75,75,128)
    x = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 8 : DWConv2d + BN
    # (75,75,128) => (38,38,128)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 9 : PWConv2d + BN
    # (38,38,128) => (38,38,256)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 10 : DWConv2d + BN
    # (38,38,256) => (38,38,256)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 11 : PWConv2d + BN
    # (38,38,256) => (38,38,256)
    x = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 12 : DWConv2d + BN
    # (38,38,256) => (19,19,256)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 13 : PWConv2d + BN
    # (19,19,256) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # Same 5 times DW + PW : 1
    # layer 14 : DWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 15 : PWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # Same 5 times DW + PW : 2
    # layer 16 : DWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 17 : PWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # Same 5 times DW + PW : 3
    # layer 18 : DWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 19 : PWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # Same 5 times DW + PW : 4
    # layer 20 : DWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 21 : PWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # Same 5 times DW + PW : 5
    # layer 22 : DWConv2d + BN
    # (19,19,512) => (19,19,512)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer 23 : PWConv2d + BN
    # (19,19,512) => (19,19,512)
    conv4 = tf.keras.layers.Conv2D(512, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(x)
    # x = keras.layers.BatchNormalization(fused=False)(x)

    # layer feature 1 :
    # (19,19,512) => (19,19, 4 * (class_num + 4))
    f1 = tf.keras.layers.Conv2D((4 * (class_num + 4)), kernel_size=3, activation="relu", padding="same",
                                use_bias=False)(conv4)
    f1 = tf.keras.layers.Reshape((19 * 19 * 4, (class_num + 4)), input_shape=(19, 19, 4 * (class_num + 4)))(f1)

    # layer 24,25 : Conv2d + PWConv2d
    # (19,19,512) => (10,10,1024) => (10,10,1024)
    fc6 = tf.keras.layers.Conv2D(1024, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(
        conv4)

    fc7 = tf.keras.layers.Conv2D(1024, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(fc6)

    # layer feature 2 :
    # (10,10,1024) => (10, 10, 6 * (class_num + 4))
    f2 = tf.keras.layers.Conv2D((6 * (class_num + 4)), kernel_size=3, activation="relu", padding="same",
                                use_bias=False)(fc7)
    f2 = tf.keras.layers.Reshape((10 * 10 * 6, (class_num + 4)), input_shape=(10, 10, 6 * (class_num + 4)))(f2)

    # layer 26,27 : PWConv2d + Conv2d
    # (10,10,1024) => (10,10,256) => (5,5,512)
    conv8_1 = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(
        fc7)
    conv8_2 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(
        conv8_1)

    # layer feature 3 :
    # (5,5,512) => (5, 5, 6 * (class_num + 4))
    f3 = tf.keras.layers.Conv2D((6 * (class_num + 4)), kernel_size=3, activation="relu", padding="same",
                                use_bias=False)(conv8_2)
    f3 = tf.keras.layers.Reshape((5 * 5 * 6, (class_num + 4)), input_shape=(5, 5, 6 * (class_num + 4)))(f3)

    # layer 28,29 : PWConv2d + Conv2d
    # (5,5,512) => (5,5,128) => (3,3,256)
    conv9_1 = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(
        conv8_2)
    conv9_2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, activation="relu", padding="same", use_bias=False)(
        conv9_1)

    # layer feature 4 :
    # (3,3,256) => (3, 3, 4 * (class_num + 4))
    f4 = tf.keras.layers.Conv2D((4 * (class_num + 4)), kernel_size=3, activation="relu", padding="same",
                                use_bias=False)(conv9_2)
    f4 = tf.keras.layers.Reshape((3 * 3 * 4, (class_num + 4)), input_shape=(3, 3, 4 * (class_num + 4)))(f4)

    # layer 30,31 : PWConv2d + Conv2d
    # (3,3,256) => (3,3,128) => (1,1,256)
    conv10_1 = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, activation="relu", padding="same", use_bias=False)(
        conv9_2)
    conv10_2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same", use_bias=False)(
        conv10_1)

    # layer feature 5 :
    # (3,3,256) => (1, 1, 4 * (class_num + 4))
    f5 = tf.keras.layers.Conv2D((4 * (class_num + 4)), kernel_size=3, activation="relu", use_bias=False)(conv10_2)
    f5 = tf.keras.layers.Reshape((1 * 1 * 4, (class_num + 4)), input_shape=(1, 1, 4 * (class_num + 4)))(f5)

    # Concat

    # 4 6 6 6 4 4
    conv4_mbox_loc = tf.keras.layers.Conv2D(4 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                            kernel_regularizer=l2(l2_reg), name='conv4_mbox_loc')(conv4)
    # conv4_mbox_loc_flat = Flatten()(conv4_mbox_loc)

    conv4_mbox_conf = tf.keras.layers.Conv2D(4 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                             kernel_regularizer=l2(l2_reg), name='conv4_mbox_conf')(conv4)
    # conv4_mbox_conf_flat = Flatten()(conv4_mbox_conf)

    # 6
    fc7_mbox_loc = tf.keras.layers.Conv2D(6 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                          kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    # f7_mbox_loc_flat = Flatten()(f7_mbox_loc)

    fc7_mbox_conf = tf.keras.layers.Conv2D(6 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                           kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    # f7_mbox_conf_flat = Flatten()(f7_mbox_conf)

    # 6
    conv8_2_mbox_loc = tf.keras.layers.Conv2D(6 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    # conv8_2_mbox_loc_flat = Flatten()(conv8_2_mbox_loc)

    conv8_2_mbox_conf = tf.keras.layers.Conv2D(6 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                               kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    # conv8_2_mbox_conf_flat = Flatten()(conv8_2_mbox_conf)

    # 6
    conv9_2_mbox_loc = tf.keras.layers.Conv2D(6 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                              kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)
    # conv9_2_mbox_loc_flat = Flatten()(conv9_2_mbox_loc)

    conv9_2_mbox_conf = tf.keras.layers.Conv2D(6 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                               kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
    # conv9_2_mbox_conf_flat = Flatten()(conv9_2_mbox_conf)

    # 4

    conv10_2_mbox_loc = tf.keras.layers.Conv2D(4 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                               kernel_regularizer=l2(l2_reg), name='conv10_2_mbox_loc')(conv10_2)
    # conv10_2_mbox_loc_flat = Flatten()(conv10_2_mbox_loc)

    conv10_2_mbox_conf = tf.keras.layers.Conv2D(4 * 4, 3, 3, padding='same', kernel_initializer='he_normal',
                                                kernel_regularizer=l2(l2_reg), name='conv10_2_mbox_conf')(conv10_2)
    # conv10_2_mbox_conf_flat = Flatten()(conv10_2_mbox_conf)

    # priorbox_pool_6_temp = PriorBox((300,300), 276.0, max_size=330.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2])

    ######

    # TODO
    # AnchorBox   --> PriorBox ?? // CHECK !!

    conv4_priorbox = PriorBox.PriorBox((300, 300), 222.0, max_size=276.0, aspect_ratios=[2, 3],
                                       variances=[0.1, 0.1, 0.2, 0.2])(conv4)
    fc7_priorbox = PriorBox.PriorBox((300, 300), 222.0, max_size=276.0, aspect_ratios=[2, 3],
                                     variances=[0.1, 0.1, 0.2, 0.2])(fc7)
    conv8_2_priorbox = PriorBox.PriorBox((300, 300), 222.0, max_size=276.0, aspect_ratios=[2, 3],
                                         variances=[0.1, 0.1, 0.2, 0.2])(conv8_2)
    conv9_2_priorbox = PriorBox.PriorBox((300, 300), 222.0, max_size=276.0, aspect_ratios=[2, 3],
                                         variances=[0.1, 0.1, 0.2, 0.2])(conv9_2)
    conv10_2_priorbox = PriorBox.PriorBox((300, 300), 222.0, max_size=276.0, aspect_ratios=[2, 3],
                                          variances=[0.1, 0.1, 0.2, 0.2])(conv10_2)


    # TODO
    # Reshape

    # TODO
    # Concatenate  // conf ( batch, n_boxes_total, n_classes ) , loc ( batch, n_boxes_total, 4 ) , priorbox ( batch, n_boxes_total, 8 )

    # TODO
    # ActivationFunction

    # TODO
    # Concatenate // prediction (batch, n_boxes_total, n_classes + 4 + 8)

    z = tf.keras.layers.concatenate([f1, f2, f3, f4, f5], axis=1)
    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dense(10, activation='softmax')(z)

    ### for SSD 5 features contract

    return tf.keras.Model(inputs=inputs, outputs=z)


# train
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
    train_model = build_keras_model()

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.global_variables_initializer())

    # sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    # adaBound = AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, weight_decay=0.0, amsbound=False)

    train_model.compile(
        optimizer='adam',
        # loss='sparse_categorical_crossentropy',
        loss=ssd_loss.compute_loss,
        metrics=['accuracy']
    )
    print(train_images.shape, train_labels.shape)
    train_model.fit(train_images, train_labels, epochs=1)

    # save graph and checkpoints
    saver = tf.train.Saver()
    saver.save(train_sess, './checkpoints')

with train_graph.as_default():
    print('sample result of original model')
    print(train_model.predict(test_images[:1]))

# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    keras.backend.set_learning_phase(0)
    eval_model = build_keras_model()
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'checkpoints')

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )

    with open('./frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph('./frozen_model.pb', input_arrays=['Input'],
                                                              output_arrays=['dense/Softmax'])

converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {"Input": (0, 255)}
converter.default_ranges_stats = [0, 255]

tflite_model = converter.convert()

open("./tflite_model.tflite", "wb").write(tflite_model)



