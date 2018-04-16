"""Inception-ResNet V1 model for Keras.

Reference
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning - https://arxiv.org/abs/1602.07261
"""

import numpy as np
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import add
from keras.models import Model


class InceptionResNetV1(Model):
    def __init__(self, input_shape=(160, 160, 3), dropout_keep_prob=0.8, bottleneck_layer_size=128):
        """Initialize the Inception-ResNet-v1 model.

        :param input_shape: a 3-D tensor of size [height, width, 3].
        :param dropout_keep_prob: the fraction to keep before final layer.
        :param bottleneck_layer_size: number of predicted classes.
        """
        self._input_shape = input_shape
        self.dropout_keep_prob = dropout_keep_prob
        self.bottleneck_layer_size = bottleneck_layer_size

        super().__init__(*self.__build(), name='inception_resnet_v1')

    def _stem(self):
        """Builds the Stem of the Inception-ResNet network"""
        inputs = Input(shape=self._input_shape)
        # 149 x 149 x 32
        x = self._conv2d(inputs, 32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
        # 147 x 147 x 32
        x = self._conv2d(x, 32, 3, padding='valid', name='Conv2d_2a_3x3')
        # 147 x 147 x 64
        x = self._conv2d(x, 64, 3, name='Conv2d_2b_3x3')
        # 73 x 73 x 64
        x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)
        # 73 x 73 x 80
        x = self._conv2d(x, 80, 1, padding='valid', name='Conv2d_3b_1x1')
        # 71 x 71 x 192
        x = self._conv2d(x, 192, 3, padding='valid', name='Conv2d_4a_3x3')
        # 35 x 35 x 256
        x = self._conv2d(x, 256, 3, strides=2, padding='valid', name='Conv2d_4b_3x3')
        return inputs, x

    def _block35(self, x, block_idx, scale=1.0, activation='relu'):
        """Builds the 35x35 Inception-ResNet-A module"""
        prefix = f'Block35_{block_idx}' if block_idx is not None else None

        branch_0 = self._conv2d(x, 32, 1, name=f'{prefix}_Branch_0_Conv2d_1x1')
        branch_1 = self._conv2d(x, 32, 1, name=f'{prefix}_Branch_1_Conv2d_0a_1x1')
        branch_1 = self._conv2d(branch_1, 32, 3, name=f'{prefix}_Branch_1_Conv2d_0b_3x3')
        branch_2 = self._conv2d(x, 32, 1, name=f'{prefix}_Branch_2_Conv2d_0a_1x1')
        branch_2 = self._conv2d(branch_2, 32, 3, name=f'{prefix}_Branch_2_Conv2d_0b_3x3')
        branch_2 = self._conv2d(branch_2, 32, 3, name=f'{prefix}_Branch_2_Conv2d_0c_3x3')
        branches = [branch_0, branch_1, branch_2]

        return self.__resnet_block(x, branches, scale, activation, prefix)

    def _block17(self, x, block_idx, scale=1.0, activation='relu'):
        """Builds the 17x17 Inception-ResNet-B module"""
        prefix = f'Block17_{block_idx}' if block_idx is not None else None

        branch_0 = self._conv2d(x, 128, 1, name=f'{prefix}_Branch_0_Conv2d_1x1')
        branch_1 = self._conv2d(x, 128, 1, name=f'{prefix}_Branch_1_Conv2d_0a_1x1')
        branch_1 = self._conv2d(branch_1, 128, [1, 7], name=f'{prefix}_Branch_1_Conv2d_0b_1x7')
        branch_1 = self._conv2d(branch_1, 128, [7, 1], name=f'{prefix}_Branch_1_Conv2d_0c_7x1')
        branches = [branch_0, branch_1]

        return self.__resnet_block(x, branches, scale, activation, prefix)

    def _block8(self, x, block_idx, scale=1.0, activation='relu'):
        """Builds the 8x8 Inception-ResNet-C module"""
        prefix = f'Block8_{block_idx}' if block_idx is not None else None

        branch_0 = self._conv2d(x, 192, 1, name=f'{prefix}_Branch_0_Conv2d_1x1')
        branch_1 = self._conv2d(x, 192, 1, name=f'{prefix}_Branch_1_Conv2d_0a_1x1')
        branch_1 = self._conv2d(branch_1, 192, [1, 3], name=f'{prefix}_Branch_1_Conv2d_0b_1x3')
        branch_1 = self._conv2d(branch_1, 192, [3, 1], name=f'{prefix}_Branch_1_Conv2d_0c_3x1')
        branches = [branch_0, branch_1]

        return self.__resnet_block(x, branches, scale, activation, prefix)

    def __resnet_block(self, x, branches, scale, activation, prefix):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        mixed = Concatenate(axis=channel_axis, name=f'{prefix}_Concatenate')(branches)
        up = self._conv2d(mixed, K.int_shape(x)[channel_axis], 1, activation=None, use_bias=True,
                          name=f'{prefix}_Conv2d_1x1')

        up = Lambda(lambda _up: _up * scale, output_shape=K.int_shape(up)[1:])(up)
        x = add([x, up])
        if activation is not None:
            x = Activation(activation, name=f'{prefix}_Activation')(x)
        return x

    @staticmethod
    def _conv2d(x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)(x)
        if not use_bias:
            normalization_axis = 1 if K.image_data_format() == 'channels_first' else 3
            x = BatchNormalization(
                axis=normalization_axis,
                momentum=0.995,
                epsilon=0.001,
                scale=False,
                name=f'{name}_BatchNorm' if name else None
            )(x)
        if activation is not None:
            x = Activation(activation, name=f'{name}_Activation' if name else None)(x)
        return x

    def _reduction_a(self, x, k=192, l=192, m=256, n=384):
        """Builds the 35x35 to 17x17 Reduction-A module"""
        branch_0 = self._conv2d(x, n, 3, strides=2, padding='valid', name=f'Mixed_6a_Branch_0_Conv2d_1a_3x3')
        branch_1 = self._conv2d(x, k, 1, name=f'Mixed_6a_Branch_1_Conv2d_0a_1x1')
        branch_1 = self._conv2d(branch_1, l, 3, name=f'Mixed_6a_Branch_1_Conv2d_0b_3x3')
        branch_1 = self._conv2d(branch_1, m, 3, strides=2, padding='valid', name=f'Mixed_6a_Branch_1_Conv2d_1a_3x3')
        branch_pool = MaxPooling2D(3, strides=2, padding='valid', name=f'Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_pool]
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        return Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    def _reduction_b(self, x):
        """Builds the 17x17 to 8x8 Reduction-B module"""
        branch_0 = self._conv2d(x, 256, 1, name=f'Mixed_7a_Branch_0_Conv2d_0a_1x1')
        branch_0 = self._conv2d(branch_0, 384, 3, strides=2, padding='valid', name=f'Mixed_7a_Branch_0_Conv2d_1a_3x3')
        branch_1 = self._conv2d(x, 256, 1, name=f'Mixed_7a_Branch_1_Conv2d_0a_1x1')
        branch_1 = self._conv2d(branch_1, 256, 3, strides=2, padding='valid', name=f'Mixed_7a_Branch_1_Conv2d_1a_3x3')
        branch_2 = self._conv2d(x, 256, 1, name=f'Mixed_7a_Branch_2_Conv2d_0a_1x1')
        branch_2 = self._conv2d(branch_2, 256, 3, name=f'Mixed_7a_Branch_2_Conv2d_0b_3x3')
        branch_2 = self._conv2d(branch_2, 256, 3, strides=2, padding='valid', name=f'Mixed_7a_Branch_2_Conv2d_1a_3x3')
        branch_3_pool = MaxPooling2D(3, strides=2, padding='valid', name=f'Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_2, branch_3_pool]
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        return Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    def __build(self):
        """Builds the Inception-ResNet-v1 model"""
        inputs, x = self._stem()

        # 5 x Inception-ResNet-A module
        for block_idx in range(1, 6):
            x = self._block35(x, block_idx, scale=0.17)

        # Reduction-A module
        x = self._reduction_a(x)

        # 10 x Inception-ResNet-B module
        for block_idx in range(1, 11):
            x = self._block17(x, block_idx, scale=0.1)

        # Reduction-B module
        x = self._reduction_b(x)

        # 5 x Inception-ResNet-C module
        for block_idx in range(1, 6):
            x = self._block8(x, block_idx, scale=0.2)
        x = self._block8(x, block_idx=6, activation=None)

        x = GlobalAveragePooling2D(name='AvgPool')(x)
        x = Dropout(1.0 - self.dropout_keep_prob, name='Dropout')(x)

        # Bottleneck
        x = Dense(self.bottleneck_layer_size, use_bias=False, name='Bottleneck')(x)
        x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)

        return inputs, x

    def predict(self, x, batch_size=None, verbose=0, steps=None, l2_normalize=True):
        prediction = super().predict(x, batch_size, verbose, steps)
        return self.l2_normalize(prediction) if l2_normalize else prediction

    @staticmethod
    def l2_normalize(x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output
