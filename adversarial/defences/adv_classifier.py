import tensorflow as tf
from tensorflow.keras import layers, models


class AdvClassifier(models.Model):

    def __init__(self):
        super(AdvClassifier, self).__init__()

        # model structures here
        self.conv1 = tf.keras.layers.Conv2D(filters=100 , stride = 2, kernel_size = (3,3) ,activation=tf.nn.leaky_relu)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=100 , stride = 2, kernel_size = (3,3) , activation=tf.nn.leaky_relu)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=100 , stride = 2, kernel_size = (2,2) , activation=tf.nn.leaky_relu)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(20, activation = tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(2, activation = 'softmax')

    def call(self, x, training = True):
        # forward propagation here
        x = self.conv1(x, training = training)
        x = self.bn1(x,training = training)
        x = self.conv2(x,training = training)
        x = self.bn2(x,training = training)
        x = self.conv3(x,training = training)
        x = self.bn3(x,training = training)
        x = self.flatten(x)
        x = self.dense1(x,training = training)
        return self.dense2(x,training = training)
    
    def load_custom_weights(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.set_weights(data)
            
    def save_custom_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_weights(), f)
        
