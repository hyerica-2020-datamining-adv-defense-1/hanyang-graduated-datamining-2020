import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class EnsembleModel(models.Model):

    def __init__(self, model_cls_list, devices):
        super(EnsembleModel, self).__init__()
        self.models = []
        
        n_devices = len(devices)
        
        for i, model_cls in enumerate(model_cls_list):
            with tf.device(f"/gpu:{devices[i % n_devices]}"):
                self.models.append(model_cls())

    def call(self, inputs, training=False):
        model = np.random.choice(self.models)
        return model(inputs, training=training)
