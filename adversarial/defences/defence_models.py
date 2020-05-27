import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, applications


class BaseModel(models.Model):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.base_model = applications.MobileNetV2(input_shape=[160, 160, 3], include_top=False, weights=None)
        self.top_layer = models.Sequential([
            layers.Dense(10),
            layers.Activation(tf.nn.softmax),
        ])

    def load_custom_weights(self, path):
        self.build(input_shape=(None, 160, 160, 3))

        with open(path, "rb") as f:
            weights = pickle.load(f)
            self.set_weights(weights)
        
    def call(self,inputs,training=False):
        x = self.base_model(inputs, training=training)
        x = layers.Flatten()(x)
        outputs = self.top_layer(x, training=training)

        return outputs