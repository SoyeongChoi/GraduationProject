from tensorflow.keras import layers, Model


class MobilenetSSD(object):

    def __init__(self, input_shape=(300, 300, 3), num_classes=6, only_mobilenet=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.only_mobilenet = only_mobilenet

    def make_model(self):
        inputs = layers.Input(shape=self.input_shape, name="Input")

        # layer 1 : Conv2d + BN
        # (300,300,3) => (150,150,32)
        x = layers.Conv2D(32, kernel_size=3, strides=2, activation=None, padding="same", use_bias=False)(
            inputs)
        x = layers.BatchNormalization(fused=False)(x)
        x = layers.ReLU(6.)(x)

        # depthwise_separable_convolution(self, input_layer, output_channels, strides=(1, 1)):
        x = depthwise_separable_convolution(x, 64)
        x = depthwise_separable_convolution(x, 128, (2, 1))
        x = depthwise_separable_convolution(x, 128)
        x = depthwise_separable_convolution(x, 256, (2, 1))
        x = depthwise_separable_convolution(x, 256)
        x = depthwise_separable_convolution(x, 512, (2, 1))

        x = depthwise_separable_convolution(x, 512)
        x = depthwise_separable_convolution(x, 512)
        x = depthwise_separable_convolution(x, 512)
        x = depthwise_separable_convolution(x, 512)
        x = depthwise_separable_convolution(x, 512)

        if self.only_mobilenet == True:
            x = depthwise_separable_convolution(x, 1024, (2, 1))
            x = depthwise_separable_convolution(x, 1024)
            x = layers.AveragePooling2D(pool_size=(7, 7))(x)

            x = layers.Dense(self.num_classes)(x)

            outputs = layers.Activation('sigmoid', name='Output')(x)

            return Model(inputs=inputs, outputs=outputs)
        # for_ssd
        else:
            return inputs, x


def depthwise_separable_convolution(input_layer, output_channels, strides=(1, 1)):
    # layer : DWConv2d + BN + ReLU6
    x = layers.DepthwiseConv2D(kernel_size=3, strides=strides[0], activation=None, padding="same", use_bias=False)(
        input_layer)
    x = layers.BatchNormalization(fused=False)(x)
    x = layers.ReLU(6.)(x)

    # layer : PWConv2d + BN + ReLU6
    x = layers.Conv2D(output_channels, kernel_size=1, strides=strides[1], activation=None, padding="same",
                      use_bias=False)(x)
    x = layers.BatchNormalization(fused=False)(x)
    output = layers.ReLU(6.)(x)

    return output
