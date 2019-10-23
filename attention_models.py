import tensorflow as tf

class WeightedAverageAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WeightedAverageAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(WeightedAverageAttention, self).build(input_shape)
  
    def call(self, inputs):
        weights = inputs[0]
        inputs = inputs[1]
        
        # return tf.reduce_mean(tf.expand_dims(weights, axis=-1)*inputs, axis=-2)
        return tf.reduce_mean(weights*inputs, axis=-2)
        # return tf.reduce_sum(inputs*weights, axis=-2)/(tf.reduce_sum(weights, axis=-2) + 0.00000001)
  
    def compute_output_shape(self, input_shape):
        return [input_shape[1,0], input_shape[1,2]]
  
    def get_config(self):
        base_config = super(WeightedAverageAttention, self).get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class WeightedAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        weights = inputs[0]
        inputs = inputs[1]
        
        return inputs*weights
  
    def compute_output_shape(self, input_shape):
        return [input_shape[1,0], input_shape[1,1]*input_shape[1,2]]
  
    def get_config(self):
        base_config = super().get_config()
        return base_config
  
    @classmethod
    def from_config(cls, config):
        return cls(**config)