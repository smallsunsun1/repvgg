import tensorflow as tf
import numpy as np
from tensorflow import keras
from typing import Union


class ConvBN(keras.layers.Layer):
    def __init__(self, out_channels: int, kernel_size: Union(int, tuple), stride: Union(int, tuple), groups: int, padding: str, **kwargs):
        super(ConvBN, self).__init__(**kwargs)
        self.conv = keras.layers.Conv2D(
            out_channels, kernel_size, stride, padding=padding, use_bias=False, groups=groups)
        self.bn = keras.layers.BatchNormalization()
        

    def call(self, x, **kwargs):
        x = self.conv(x, **kwargs)
        x = self.bn(x, **kwargs)
        return x


class RepVGGBlock(keras.layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, stride: Union(int, tuple), dialation=1, groups=1, delopy=False, **kwargs):
        super(RepVGGBlock, self).__init__(**kwargs)
        self.delopy = delopy
        self.in_channels = in_channels
        self.groups = groups
        if self.delopy:
            self.rbr_reparam = keras.layers.Conv2D(out_channels, 3, stride, padding="same", dilation_rate=dialation,
                                                    groups=groups, use_bias=True)
        else:
            self.rbr_identity = keras.layers.BatchNormalization() if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvBN(out_channels, (3, 3), stride,padding="same", groups=groups)
            self.rbr_1x1 = ConvBN(out_channels, kernel_size=(1, 1), stride=stride, padding="valid")
        self.nonlinearity = keras.layers.ReLU()
    def call(self, x, **kwargs): 
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(x, **kwargs), **kwargs)
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x, **kwargs)
        return self.nonlinearity(self.rbr_dense(x, **kwargs) + self.rbr_1x1(x, **kwargs) + id_out)
    def get_equivalent_kernel_bias(self):
        pass
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        elif isinstance(branch, ConvBN):
            kernel = branch.conv.weights
            running_mean = branch.bn.moving_mean
            running_var = branch.bn.moving_variance
            gamma = branch.bn.gamma
            beta = branch.bn.beta
            eps = branch.bn.epsilon
        elif isinstance(branch, keras.layers.BatchNormalization):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, 3, 3, input_dim), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i] = 1
        else:
            raise NotImplementedError("Current Type Of Branch Are Not Supported!")

