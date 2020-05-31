import tensorflow as tf
from tensorflow.keras import models, layers


class FeatureDistinctionBlock(models.Model):

    def __init__(self):
        super(FeatureDistinctionBlock, self).__init__()

        # network structure here
        

    def call(self, inputs, training=False):
        cache = dict() # cache["losses"] = ?
        
        # forward propagation here


        outputs = None # set proper value
        return outputs, cache
