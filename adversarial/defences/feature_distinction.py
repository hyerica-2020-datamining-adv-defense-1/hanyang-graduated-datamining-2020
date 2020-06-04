import tensorflow as tf
from tensorflow.keras import models, layers


class FeatureDistinctionBlock(models.Model):

    def __init__(self, feat_dim):
        super(FeatureDistinctionBlock, self).__init__()
        self.num_classes = 10
        self.feat_dim = feat_dim
        
        self.centers = tf.Variable(tf.random.normal((self.num_classes,self.feat_dim)), trainable = True)

    def call(self, inputs, labels, training=False):

        batch_size = tf.shape(inputs)[0]
        
        distance_matrix = tf.tile(tf.reduce_sum(inputs**2 , axis=1, keepdims = True), (1,self.num_classes)) + \
            tf.transpose(tf.tile(tf.reduce_sum(self.centers**2, axis=1, keepdims = True), (1, batch_size)), (1,0))
        distance_matrix = distance_matrix - 2*(tf.matmul(inputs, tf.transpose(self.centers, (1,0))))

        classes = tf.range(self.num_classes, dtype=tf.float32)
        labels = tf.tile(tf.expand_dims(labels,1),(1,self.num_classes))

        mask1 = tf.cast(labels, dtype=tf.int32) == tf.cast(tf.tile(tf.expand_dims(classes,0), (batch_size,1)), dtype=tf.int32)
        mask2 = tf.cast(labels, dtype=tf.int32) != tf.cast(tf.tile(tf.expand_dims(classes,0), (batch_size,1)), dtype=tf.int32)

        pos_distance = distance_matrix * tf.cast(mask1, dtype=tf.float32)
        pos_distance = tf.clip_by_value(pos_distance,1e-12,1e+12)
        
        neg_distance = distance_matrix * tf.cast(mask2, dtype=tf.float32)
        neg_distance = tf.clip_by_value(neg_distance, 1e-12, 1e+12)

        loss = tf.reduce_mean(pos_distance)/(tf.reduce_mean(neg_distance)/(self.num_classes - 1) + 1e-8)
        
        return loss

