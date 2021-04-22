import copy
import tensorflow as tf
import numpy as np
from tensorflow import keras
from typing import Union


class ConvBN(keras.layers.Layer):
    def __init__(self, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple], groups: int, padding: str, **kwargs):
        super(ConvBN, self).__init__(**kwargs)
        self.conv = keras.layers.Conv2D(
            out_channels, kernel_size, stride, padding=padding, use_bias=False, groups=groups)
        self.bn = keras.layers.BatchNormalization()

    def call(self, x, **kwargs):
        x = self.conv(x, **kwargs)
        x = self.bn(x, **kwargs)
        return x


class RepVGGBlock(keras.layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, stride: Union[int, tuple], dialation=1, groups=1, deploy=False, **kwargs):
        super(RepVGGBlock, self).__init__(**kwargs)
        self.deploy = deploy
        self.in_channels = in_channels
        self.groups = groups
        if self.deploy:
            self.rbr_reparam = keras.layers.Conv2D(out_channels, 3, stride, padding="same", dilation_rate=dialation,
                                                   groups=groups, use_bias=True)
        else:
            self.rbr_identity = keras.layers.BatchNormalization(
            ) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvBN(
                out_channels, (3, 3), stride, padding="same", groups=groups)
            self.rbr_1x1 = ConvBN(out_channels, kernel_size=(
                1, 1), stride=stride, padding="valid", groups=groups)
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
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(kernel1x1, [[1, 1], [1, 1], [0, 0], [0, 0]], constant_values=0)

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
                kernel_value = np.zeros(
                    (3, 3, self.in_channels, input_dim), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[:, :, i, i % input_dim] = 1
                self.id_tensor = tf.convert_to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.moving_mean
            running_var = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        else:
            raise NotImplementedError(
                "Current Type Of Branch Are Not Supported!")
        std = tf.sqrt(running_var + eps)
        t = tf.reshape(gamma / std, [-1, 1, 1, 1])
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = keras.layers.Conv2D(filters=self.rbr_dense.conv.filters,
                                               kernel_size=self.rbr_dense.conv.kernel_size, strides=self.rbr_dense.conv.strides,
                                               padding=self.rbr_dense.conv.padding, dilation_rate=self.rbr_dense.conv.dilation_rate,
                                               groups=self.rbr_dense.conv.groups, use_bias=True)
        self.rbr_reparam.set_weights([kernel, bias])
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')


class RepVGGModule(keras.models.Model):
    def __init__(self, num_blocks: int, num_classes: int, width_multiplier=None,
                 override_groups_map=None, deploy=False, **kwargs):
        super(RepVGGModule, self).__init__(**kwargs)
        assert len(
            width_multiplier) == 4, "width multiplier must be multiple of 4!"
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map, "0 must be in self.override_groups_map"
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.cur_layer_idx = 1
        self.stage0 = RepVGGBlock(
            in_channels=3, out_channels=self.in_planes, stride=2, deploy=self.deploy)
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = keras.layers.GlobalAveragePooling2D()
        self.linear = keras.layers.Dense(num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes,
                          stride=stride, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return keras.models.Sequential(blocks)

    def call(self, x, **kwargs):
        out = self.stage0(x, **kwargs)
        out = self.stage1(out, **kwargs)
        out = self.stage2(out, **kwargs)
        out = self.stage3(out, **kwargs)
        out = self.stage4(out, **kwargs)
        out = self.gap(out, **kwargs)
        out = tf.reshape(out, (tf.shape(out)[0], -1))
        out = self.linear(out, **kwargs)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False):
    return RepVGGModule(num_blocks=[2, 4, 14, 1], num_classes=1000,
                        width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A1(deploy=False):
    return RepVGGModule(num_blocks=[2, 4, 14, 1], num_classes=1000,
                        width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A2(deploy=False):
    return RepVGGModule(num_blocks=[2, 4, 14, 1], num_classes=1000,
                        width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def create_RepVGG_B0(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1g2(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B2g2(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B3g2(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B3g4(deploy=False):
    return RepVGGModule(num_blocks=[4, 6, 16, 1], num_classes=1000,
                        width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]

def repvgg_model_convert(model: keras.models.Model, save_path=None, do_copy=False):
    if do_copy:
        model = copy.deepcopy(model)
    for layer in model.layers:
        if hasattr(layer, "switch_to_deploy"):
            layer.switch_to_deploy()
    if save_path is not None:
        tf.saved_model.save(model, save_path)
    return model
