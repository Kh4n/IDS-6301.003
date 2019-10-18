import tensorflow as tf

class WeightedAverageAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedAverageAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(WeightedAverageAttention, self).build(input_shape)
  
    def call(self, inputs):
        weights = inputs[1]
        inputs = inputs[0]
        
        return tf.reduce_mean(tf.expand_dims(weights, axis=-1)*inputs, axis=-2)
  
    def compute_output_shape(self, input_shape):
        return input_shape[0][2]
  
    def get_config(self):
        base_config = super(WeightedAverageAttention, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)