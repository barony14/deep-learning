from keras.layers import *
from keras.activations import relu, softmax
from keras_layer_normalization import LayerNormalization
import tensorflow as tf


def global_context_block(x):
    """
    高维矩阵乘法[n,1,c,hw]*[n,1,hw,1]=[n,1,c,1]
    GC_block:global context block
    :parameter x:input layers or tensor
    """

    bs, h, w, c = x.get_shape().as_list()
    input_x = x
    input_x = Reshape((-1, c))(input_x)  # [N, H*W, C]
    input_x = tf.transpose(input_x, perm=[0, 2, 1])  # [N,C,H*W]
    input_x = tf.expand_dims(input_x, axis=1)

    context_mask = Conv2D(filters=1, kernel_size=(1, 1))(x)
    context_mask = Reshape((-1, 1))(context_mask)
    context_mask = softmax(context_mask, axis=1)  # [N, H*W, 1]
    context_mask = tf.transpose(context_mask, [0, 2, 1])
    context_mask = tf.expand_dims(context_mask, axis=-1)

    context = tf.matmul(input_x, context_mask)  # [N,1,c,1]
    context = Reshape((1, 1, c))(context)

    context_transform = Conv2D(int(c/8), (1, 1))(context)
    context_transform = LayerNormalization()(context_transform)
    context_transform = relu(context_transform)
    context_transform = Conv2D(c, (1, 1))(context_transform)

    x = x + context_transform

    return x
