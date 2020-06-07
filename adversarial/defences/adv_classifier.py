import tensorflow as tf
from tensorflow.keras import layers, models, applications
import pickle


class AdvClassifier(models.Model):

    def __init__(self):
        super(AdvClassifier, self).__init__()

        # model structures here
        self.top_layer = models.Sequential([
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Conv2D(32, (3, 3), padding="same"),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            layers.Flatten(),
            layers.Dense(512),
            layers.Activation("relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(2),
            layers.Activation("softmax")])

    def call(self, x, training = True):
        # forward propagation here
        outputs = self.top_layer(x, training = True)
        return outputs
    
    def load_custom_weights(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.set_weights(data)
            
    def save_custom_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_weights(), f)
        

class AdvClassifierMk2(models.Model):

    def __init__(self):
        super(AdvClassifierMk2, self).__init__()
        self.base_model = applications.VGG16(weights='imagenet', include_top = False)
        #self.base_model.trainable = False
        self.top_layer = models.Sequential([
            layers.Dense(20),
            layers.Activation(tf.nn.leaky_relu),
            layers.Dropout(0.5),
            layers.Dense(2),
            layers.Activation(tf.nn.softmax),
        ])

    def load_custom_weights(self, path):
        self.build(input_shape=(None, 160, 160, 3))

        with open(path, "rb") as f:
            weights = pickle.load(f)
            self.set_weights(weights)

    def save_custom_weights(self, path):
        with open(path, "wb") as f:
            weights = self.get_weights()
            pickle.dump(weights, f)
        
    def call(self,inputs,training=False):
        x = self.base_model(inputs, training=False)
        x = layers.Flatten()(x)
        outputs = self.top_layer(x, training=training)

        return outputs