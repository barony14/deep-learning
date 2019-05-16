from keras.layers import *
from keras.activations import relu
from keras_layer_normalization import LayerNormalization
import tensorflow as tf


def global_context_block(x):
    bs, h, w, c = x.get_shape().as_list()
    input_x = x
    input_x = hw_flatten(input_x)  # [N, H*W, C]
    input_x = tf.transpose(input_x, perm=[0, 2, 1])
    input_x=tf.reshape(input_x,[-1,784])
    #input_x = tf.expand_dims(input_x, axis=1)

    context_mask = Conv2D(filters=1,kernel_size=(1,1))(x)
    context_mask = hw_flatten(context_mask)
    context_mask = tf.nn.softmax(context_mask, axis=1)  # [N, H*W, 1]
    context_mask = tf.transpose(context_mask, perm=[1,2,0])
    context_mask=tf.reshape(context_mask,[784,-1])
    #context_mask = tf.expand_dims(context_mask, axis=2)

    context = tf.matmul(input_x, context_mask)
    context = tf.reshape(context, [-1, 1, 1, c])

    context_transform = Conv2D(c, (1,1))(context)
    context_transform = LayerNormalization()(context_transform)
    context_transform = relu(context_transform)
    context_transform = Conv2D(c,(1,1))(context_transform)

    x = x + context_transform

    return x

def hw_flatten(x):
    return tf.reshape(x, [-1, x.get_shape().as_list()[1]*x.get_shape().as_list()[2], x.get_shape().as_list()[3]])