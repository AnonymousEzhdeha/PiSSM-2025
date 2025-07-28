import tensorflow as tf
from tensorflow import keras as k


class LayerNormalizer(k.layers.Layer):


    def build(self, input_shape):
        shape = input_shape[-1:]
        # self._offset = self.add_weight("bias", shape=shape, initializer=k.initializers.zeros)
        # self._scale = self.add_weight("weight", shape=shape, initializer=k.initializers.ones)
        self._offset = self.add_weight(name="bias", shape=shape, initializer=k.initializers.zeros)
        self._scale  = self.add_weight(name="weight", shape=shape, initializer=k.initializers.ones)

    def call(self, inputs, **kwargs):
        """
        batch normalizations is a built in function in tensorflow
        """
        norm_axis = [i + 1 for i in range(len(inputs.get_shape()) - 1)]
        mean, var = tf.nn.moments(inputs, axes=norm_axis, keepdims=True)
        return tf.nn.batch_normalization(inputs,
                                         mean=mean, variance=var, variance_epsilon=1e-10,
                                         offset=self._offset, scale=self._scale)

    def compute_output_shape(self, input_shape):
        return input_shape