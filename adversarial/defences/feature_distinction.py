import tensorflow as tf
from tensorflow.keras import models, layers


class FeatureDistinctionBlock(models.Model):

    def __init__(self, feat_dim):
        super(FeatureDistinctionBlock, self).__init__()
        self.num_classes = 10
        self.feat_dim = feat_dim
        
        self.centers = tf.Variable(tf.random.normal((self.num_classes,self.feat_dim)), trainable = True)

        # network structure here        

    def call(self, inputs, labels, training=False):

        batch_size = tf.shape(inputs)[0]
        
        distance_matrix = tf.tile(tf.reduce_sum(inputs**2 , axis=1, keepdims = True), (1,self.num_classes)) + \
            tf.tile(tf.reduce_sum(self.centers**2, axis=1, keepdims = True), (1, batch_size)).transpose(1,0)

        distance_matrix = distance_matrix - 2*(tf.matmul(inputs, self.centers.transpose(1,0)))

        classes = tf.cast(tf.range(self.num_classes), tf.int32)

        labels = tf.tile(tf.expand_dims(labels,1),(1,self.num_classes))

        mask = (labels) == (tf.tile(tf.expand_dims(classes,0), (batch_size,1)))

        distance = distance_matrix * mask
        distance = tf.clip_by_value(distance,1e-12,1e+12)

        loss = tf.reduce_mean(distance_matrix)
        
        return loss
