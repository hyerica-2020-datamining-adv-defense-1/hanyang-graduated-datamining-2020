import tensorflow as tf
from tensorflow.keras import layers, models


class AdvClassifier(models.Model):

    def __init__(self):
        super(AdvClassifier, self).__init__()

        # model structures here
        self.conv1 = tf.keras.layers.Conv2D(filters=100 , kernel_size = (3,3) ,activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=100 , kernel_size = (3,3) , activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(20, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(2, activation = 'softmax')

    def call(self, x):
        # forward propagation here
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
        
