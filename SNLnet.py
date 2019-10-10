from keras import layers
from keras_layer_normalization import LayerNormalization
import tensorflow as tf
import keras.backend as K


def snl(x):
    """
    simplified non local net
    GCnet 发现在NLnet中图像每个点的全局上下文相近，只计算一个点的全局相似度，计算量减少1/hw
    :parameter x:input layers or tensor
    """

    bs, h, w, c = x.get_shape().as_list()
    input_x = x
    input_x = layers.Reshape((h*w, c))(input_x)  # [bs, H*W, C]
    # input_x = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input_x)  # [bs,C,H*W]
    # input_x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(input_x)  # [bs,1,C,H*W]

    context_mask = layers.Conv2D(filters=1, kernel_size=(1, 1))(x) # [bs,h,w,1]
    context_mask = layers.Reshape((h*w, 1))(context_mask)
    context_mask = layers.Softmax(axis=1)(context_mask)  # [bs, H*W, 1]
    # context_mask = layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(context_mask)
    # context_mask = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(context_mask)

    context = layers.dot([input_x, context_mask],axes=1)  # [bs,1,c]
    context = layers.Reshape((1, 1, c))(context)

    # context_transform = layers.Conv2D(c, (1, 1))(context)
    # context_transform = LayerNormalization()(context_transform)
    # context_transform = layers.ReLU()(context_transform)
    # context_transform = layers.Conv2D(c, (1, 1))(context_transform)

    context_transform=layers.Conv2D(c,kernel_size=(1,1))(context)
    #
    x = layers.Add()([x,context_transform])

    return x
