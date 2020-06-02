import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class EnsembleModel(models.Model):

    def __init__(self, model_cls_list):
        super(EnsembleModel, self).__init__()
        self.models = []
        
        for i, model_cls in enumerate(model_cls_list):
            self.models.append(model_cls())

    def call(self, inputs, training=False):
        model = np.random.choice(self.models)
        return model(inputs, training=training)
