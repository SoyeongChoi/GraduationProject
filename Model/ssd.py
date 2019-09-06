import tensorflow as tf
import mobilenet

from tensorflow.keras import layers


class SSD(object):
    def __init__(self, input_shape=(300, 300, 3),
                 num_classes=6, backbone='Mobilenet_v1',
                 prior_box_count=[3, 6, 6, 6, 6, 6], feature_map_size=[19, 10, 5, 3, 2, 1]
                 ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.prior_box_count = prior_box_count
        self.feature_map_size = feature_map_size

    def make_model(self):

        if self.backbone == 'Mobilenet_v1':

            backbone_model = mobilenet.MobilenetSSD(self.input_shape, self.num_classes)
            inputs, outputs = backbone_model.make_model()

            f1_loc, f1_conf = feature_map(outputs, self.num_classes, self.prior_box_count[0], self.feature_map_size[0])

            # input_layer, output_channels, strides=(1, 1)
            x = mobilenet.depthwise_separable_convolution(outputs, 1024, strides=(2, 1))
            x = mobilenet.depthwise_separable_convolution(x, 1024, strides=(1, 1))

            f2_loc, f2_conf = feature_map(x, self.num_classes, self.prior_box_count[1], self.feature_map_size[1])
            x = DW_PW_conv(input_layer=x, channels=(256, 512))
            f3_loc, f3_conf = feature_map(x, self.num_classes, self.prior_box_count[2], self.feature_map_size[2])
            x = DW_PW_conv(input_layer=x, channels=(128, 256))
            f4_loc, f4_conf = feature_map(x, self.num_classes, self.prior_box_count[3], self.feature_map_size[3])
            x = DW_PW_conv(input_layer=x, channels=(128, 256))
            f5_loc, f5_conf = feature_map(x, self.num_classes, self.prior_box_count[4], self.feature_map_size[4])
            x = DW_PW_conv(input_layer=x, channels=(128, 256))
            f6_loc, f6_conf = feature_map(x, self.num_classes, self.prior_box_count[5], self.feature_map_size[5])

            z_loc = layers.concatenate([f1_loc, f2_loc, f3_loc, f4_loc, f5_loc, f6_loc], axis=1, name='total_loc')

            z_conf = layers.concatenate([f1_conf, f2_conf, f3_conf, f4_conf, f5_conf, f6_conf], axis=1,
                                        name='total_conf')
            z_conf_logistic = layers.Activation('softmax')(z_conf)

            z_concat = layers.concatenate([z_conf_logistic, z_loc], name='outputs')

            return tf.keras.Model(inputs=inputs, outputs=z_concat)

        else:
            return 'other backbone is not existed!'


def feature_map(input_layer, num_classes, n_prior_box_count, n_feature_map_size):
    n2_feature_map_size = n_feature_map_size * n_feature_map_size

    feature_loc = layers.Conv2D((n_prior_box_count * (4)), kernel_size=3, activation='relu', padding="same",
                                use_bias=False)(input_layer)

    feature_conf = layers.Conv2D((n_prior_box_count * (num_classes)), kernel_size=3, activation='relu', padding="same",
                                 use_bias=False)(input_layer)

    # Reshape
    feature_loc = layers.Reshape((n2_feature_map_size * n_prior_box_count, (4)),
                                 input_shape=(n_feature_map_size, n_feature_map_size, n_prior_box_count * (4)))(
        feature_loc)
    feature_conf = layers.Reshape((n2_feature_map_size * n_prior_box_count, (num_classes)),
                                  input_shape=(
                                      n_feature_map_size, n_feature_map_size, n_prior_box_count * (num_classes)),
                                  name='f1_conf')(feature_conf)

    return feature_loc, feature_conf


def DW_PW_conv(input_layer, channels):
    x = layers.Conv2D(channels[0], kernel_size=1, strides=1, activation='relu', padding="same", use_bias=False)(
        input_layer)
    x = layers.Conv2D(channels[1], kernel_size=3, strides=2, activation='relu', padding="same", use_bias=False)(x)

    return x


