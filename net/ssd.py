import tensorflow as tf
from tensorflow.keras import layers, Model

import mobilenetv1



class SSD():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.input, self.output = mobilenetv1.mobilenet_v1(self.input_shape, self.num_classes)

    def SSD(self):
        f1_loc = layers.Conv2D((3 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(
            self.output)
        f1_loc = layers.Reshape((19 * 19 * 3, (4)), input_shape=(19, 19, 3 * (4)))(f1_loc)

        f1_conf = layers.Conv2D((3 * (self.num_classes)), kernel_size=3, activation='relu', padding="same",
                                use_bias=False)(self.output)
        f1_conf = layers.Reshape((19 * 19 * 3, (self.num_classes)), input_shape=(19, 19, 3 * (self.num_classes)),
                                 name='f1_conf')(f1_conf)

        # layer 24,25 : DWConv2d + PWConv2d
        # (19,19,512) => (10,10,512) => (10,10,1024)
        x = layers.DepthwiseConv2D(kernel_size=3, strides=2, activation=None, padding="same", use_bias=False)(
            self.backbone_output)
        x = layers.BatchNormalization(fused=False)(x)
        x = layers.ReLU(6.)(x)

        x = layers.Conv2D(1024, kernel_size=1, strides=1, activation=None, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(fused=False)(x)
        x = layers.ReLU(6.)(x)

        # layer 26,27 : DWConv2d + PWConv2d
        # (10,10,1024) => (10,10,1024) => (10,10,1024)
        x = layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(fused=False)(x)
        x = layers.ReLU(6.)(x)

        x = layers.Conv2D(1024, kernel_size=1, strides=1, activation=None, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(fused=False)(x)
        x = layers.ReLU(6.)(x)

        # layer feature 2 :
        # (10,10,1024) => (10, 10, 6 * (self.num_classes | 4))
        f2_loc = layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f2_loc = layers.Reshape((10 * 10 * 6, (4)), input_shape=(10, 10, 6 * (4)))(f2_loc)

        f2_conf = layers.Conv2D((6 * (self.num_classes)), kernel_size=3, activation='relu', padding="same",
                                use_bias=False)(x)
        f2_conf = layers.Reshape((10 * 10 * 6, (self.num_classes)), input_shape=(10, 10, 6 * (self.num_classes)),
                                 name='f2_conf')(f2_conf)

        # layer 28,29 : PWConv2d + Conv2d
        # (10,10,1024) => (10,10,256) => (5,5,512)
        x = layers.Conv2D(256, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
        x = layers.Conv2D(512, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)

        # layer feature 3 :
        # (5,5,512) => (5, 5, 6 * (self.num_classes | 4))
        f3_loc = layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f3_loc = layers.Reshape((5 * 5 * 6, (4)), input_shape=(5, 5, 6 * (4)))(f3_loc)

        f3_conf = layers.Conv2D((6 * (self.num_classes)), kernel_size=3, activation='relu', padding="same",
                                use_bias=False)(x)
        f3_conf = layers.Reshape((5 * 5 * 6, (self.num_classes)), input_shape=(5, 5, 6 * (self.num_classes)),
                                 name='f3_conf')(f3_conf)

        # layer 30,31 : PWConv2d + Conv2d
        # (5,5,512) => (5,5,128) => (3,3,256)
        x = layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
        x = layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)

        # layer feature 4 :
        # (3,3,256) => (3, 3, 4 * (self.num_classes | 4))
        f4_loc = layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding="same", use_bias=False)(x)
        f4_loc = layers.Reshape((3 * 3 * 6, (4)), input_shape=(3, 3, 6 * (4)))(f4_loc)

        f4_conf = layers.Conv2D((6 * (self.num_classes)), kernel_size=3, activation='relu', padding="same",
                                use_bias=False)(x)
        f4_conf = layers.Reshape((3 * 3 * 6, (self.num_classes)), input_shape=(3, 3, 6 * (self.num_classes)),
                                 name='f4_conf')(f4_conf)

        # layer 32,33 : PWConv2d + Conv2d
        # (3,3,256) => (3,3,128) => (2,2,256)
        x = layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
        x = layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)

        # layer feature 5 :
        # (2,2,256) => (2, 2, 4 * (self.num_classes | 4))
        f5_loc = layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
        f5_loc = layers.Reshape((2 * 2 * 6, (4)), input_shape=(1, 1, 6 * (4)))(f5_loc)

        f5_conf = layers.Conv2D((6 * (self.num_classes)), kernel_size=3, activation='relu', padding='same',
                                use_bias=False)(x)
        f5_conf = layers.Reshape((2 * 2 * 6, (self.num_classes)), input_shape=(1, 1, 6 * (self.num_classes)),
                                 name='f5_conf')(f5_conf)

        # layer 34,35 : PWConv2d + Conv2d
        # (2,2,256) => (2,2,128) => (1,1,256)
        x = layers.Conv2D(128, kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(x)
        x = layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)

        # layer feature 6 :
        # (1,1,256) => (1, 1, 4 * (self.num_classes | 4))
        f6_loc = layers.Conv2D((6 * (4)), kernel_size=3, activation='relu', padding='same', use_bias=False)(x)
        f6_loc = layers.Reshape((1 * 1 * 6, (4)), input_shape=(1, 1, 6 * (4)))(f6_loc)

        f6_conf = layers.Conv2D((6 * (self.num_classes)), kernel_size=3, activation='relu', padding='same',
                                use_bias=False)(x)
        f6_conf = layers.Reshape((1 * 1 * 6, (self.num_classes)), input_shape=(1, 1, 6 * (self.num_classes)),
                                 name='f6_conf')(f6_conf)

        z_loc = layers.concatenate([f1_loc, f2_loc, f3_loc, f4_loc, f5_loc, f6_loc], axis=1, name='total_loc')

        z_conf = layers.concatenate([f1_conf, f2_conf, f3_conf, f4_conf, f5_conf, f6_conf], axis=1, name='total_conf')
        z_conf_logistic = layers.Activation('softmax')(z_conf)

        z_concat = layers.concatenate([z_conf_logistic, z_loc], name='outputs')

        ### for SSD 5 features contract

        return Model(inputs=self.input, outputs=z_concat)
