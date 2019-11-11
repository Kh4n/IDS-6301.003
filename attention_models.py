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
    def __init__(self, scale_fac=5, **kwargs):
        self.scale_fac = scale_fac
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        weights = tf.math.tanh(self.scale_fac*inputs[0] - self.scale_fac*0.5)
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

class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, g, seq_length, **kwargs):
        self.num_heads = num_heads
        self.g = g
        self.seq_length = seq_length
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.anchors = tf.range(input_shape[1][1], dtype=tf.float32)
        self.anchors = tf.expand_dims(self.anchors, axis=0)
        self.anchors = tf.tile(self.anchors, [self.num_heads,1])
        super().build(input_shape)
    def call(self, inputs):
        aligns = self.seq_length*tf.math.sigmoid(inputs[0])
        aligns = tf.expand_dims(aligns, axis=-1)
        inputs = inputs[1]
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = tf.tile(inputs, [1,self.num_heads,1,1])
        ret = tf.expand_dims(tf.exp(-(self.anchors-aligns)**2/(2*self.g**2)), axis=-1)*inputs
        # ret = tf.math.reduce_sum(ret, axis=1)
        return ret
    def compute_output_shape(self, input_shape):
        return [input_shape[1][0], self.num_heads, input_shape[1][1], input_shape[1][2]]
    def get_config(self):
        base_config = super().get_config()
        return base_config
    @classmethod
    def from_config(cls, config):
        return cls(**config)