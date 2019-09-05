from tensorflow.keras import layers, Model


class MobileNetV1():
    def __init__(self, input_shape, num_classes, backbone=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone

    def mobilenet_v1(self):
        inputs = layers.Input(shape=self.input_shape, name="Input")
        # layer 1 : Conv2d + BN
        # (300,300,3) => (150,150,32)
        x = layers.Conv2D(32, kernel_size=3, strides=2, activation=None, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization(fused=False)(x)
        x = layers.ReLU(6.)(x)

        # layer 2 : DWConv2d + BN
        # (150,150,32) => (150,150,32)
        x = depthwise_block(x)

        # layer 3 : PWConv2d + BN
        # (150,150,32) => (150,150,64)
        x = pointwise_block(x, 64)

        # layer 4 : DWConv2d + BN
        # (150,150,64) => (75,75,64)
        x = depthwise_block(x)

        # layer 5 : PWConv2d + BN
        # (75,75,64) => (75,75,128)
        x = pointwise_block(x, 128)

        # layer 6 : DWConv2d + BN
        # (75,75,128) => (75,75,128)
        x = depthwise_block(x)

        # layer 7 : PWConv2d + BN
        # (75,75,128) => (75,75,128)
        x = pointwise_block(x, 128)

        # layer 8 : DWConv2d + BN
        # (75,75,128) => (38,38,128)
        x = depthwise_block(x)

        # layer 9 : PWConv2d + BN
        # (38,38,128) => (38,38,256)
        x = pointwise_block(x, 256)

        # layer 10 : DWConv2d + BN
        # (38,38,256) => (38,38,256)
        x = depthwise_block(x)

        # layer 11 : PWConv2d + BN
        # (38,38,256) => (38,38,256)
        x = pointwise_block(x, 256)

        # layer 12 : DWConv2d + BN
        # (38,38,256) => (19,19,256)
        x = depthwise_block(x)

        # layer 13 : PWConv2d + BN
        # (19,19,256) => (19,19,512)
        x = pointwise_block(x, 512)

        # Same 5 times DW + PW : 1
        # layer 14 : DWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = depthwise_block(x)

        # layer 15 : PWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = pointwise_block(x, 512)

        # Same 5 times DW + PW : 2
        # layer 16 : DWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = depthwise_block(x)

        # layer 17 : PWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = pointwise_block(x, 512)

        # Same 5 times DW + PW : 3
        # layer 18 : DWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = depthwise_block(x)

        # layer 19 : PWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = pointwise_block(x, 512)

        # Same 5 times DW + PW : 4
        # layer 20 : DWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = depthwise_block(x)

        # layer 21 : PWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = pointwise_block(x, 512)

        # Same 5 times DW + PW : 5
        # layer 22 : DWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = depthwise_block(x)

        # layer 23 : PWConv2d + BN
        # (19,19,512) => (19,19,512)
        x = pointwise_block(x, 512)

        if self.backbone:
            return inputs, x
        else:
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(self.num_classes, activation='softmax', use_bias=True, name='Logits')(x)
            return Model(inputs=inputs, outputs=x)


def depthwise_block(inputs):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, activation=None, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(fused=False)(x)
    x = layers.ReLU(6.)(x)

    return x


def pointwise_block(inputs, filters):
    x = layers.Conv2D(filters, kernel_size=1, strides=1, activation=None, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization(fused=False)(x)
    x = layers.ReLU(6.)(x)

    return x
