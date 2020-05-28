import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class EnsembleModel(models.Model):

    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def call(self, inputs, training=False):
        model = np.random.choice(self.models)
        return model(inputs, training=training)
