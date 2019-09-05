from tensorflow.keras import layers, Model


class MobileNetV1():
    def __init__(self, input_shape, num_classes, backbone=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone

    def mobilenet_v1(self):
        inputs = layers.Input(shape=self.input_shape, name="Input")

        # Standard Conv Block
        x = layers.Conv2D(32, kernel_size=3, strides=2, activation=None, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization(fused=False)(x)
        x = layers.ReLU(6.)(x)

        #Depthwise Separable Conv Block
        x = depthwise_separable_block(x, 64)
        x = depthwise_separable_block(x, 128)
        x = depthwise_separable_block(x, 128)
        x = depthwise_separable_block(x, 256)
        x = depthwise_separable_block(x, 256)
        x = depthwise_separable_block(x, 512)
        x = depthwise_separable_block(x, 512)
        x = depthwise_separable_block(x, 512)
        x = depthwise_separable_block(x, 512)
        x = depthwise_separable_block(x, 512)
        x = depthwise_separable_block(x, 512)

        if self.backbone:
            return inputs, x
        else:
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(self.num_classes, activation='softmax', use_bias=True, name='Logits')(x)
            return Model(inputs=inputs, outputs=x)


def depthwise_separable_block(inputs, filters):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(fused=False)(x)
    x = layers.ReLU(6.)(x)
    x = layers.Conv2D(filters, kernel_size=1, strides=1, activation=None, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(fused=False)(x)
    x = layers.ReLU(6.)(x)

    return x
