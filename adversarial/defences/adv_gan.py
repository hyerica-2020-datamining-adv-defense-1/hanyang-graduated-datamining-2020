import tensorflow as tf
from tensorflow.keras import layers, models

from adversarial.defences import AdvClassifier


class AdvGAN(models.Model):

    def __init__(self):
        super(AdvGAN, self).__init__()

        self.genereator = PerturbationGenerator()
        self.adv_clf = AdvClassifier()

    def call(self, inputs, training=False):
        n = tf.shape(inputs)[0]

        noise = tf.random.normal((n, 10, 10, 64))

        perturbation = self.generator(noise, training=training)
        clf = self.adv_clf(inputs + perturbation)

        return perturbation, clf


class PerturbationGenerator(models.Model):
    
    def __init__(self):
        super(PerturbationGenerator, self).__init__()

        self.generator = models.Sequential([
            layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="SAME", output_padding=1),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.leaky_relu),
            
            layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="SAME", output_padding=1),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.leaky_relu),
            
            layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="SAME", output_padding=1),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.leaky_relu),
            
            layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="SAME", output_padding=1),
            layers.BatchNormalization(),
            layers.Activation(tf.nn.tanh),
        ])

        self.temperature = 100

    def call(self, noise, training=False):
        x = self.generator(noise)
        x = x/self.temperature
        return x


class AdvDiscriminator(models.Model):

    def __init__(self, drop_rate=0.5):
        super(AdvDiscriminator, self).__init__()

        self.features = models.Sequential([
            layers.Conv2D(32, (5, 5), strides=(2, 2), padding="SAME"), # 80
            layers.BatchNormalization(),
            layers.Activation(tf.nn.leaky_relu),

            layers.Conv2D(32, (5, 5), strides=(2, 2), padding="SAME"), # 40
            layers.BatchNormalization(),
            layers.Activation(tf.nn.leaky_relu),

            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="SAME"), # 20
            layers.BatchNormalization(),
            layers.Activation(tf.nn.leaky_relu),

            layers.Conv2D(64, (3, 3), strides=(2, 2), padding="SAME"), # 10
            layers.BatchNormalization(),
            layers.Activation(tf.nn.leaky_relu),

            layers.Conv2D(128, (3, 3), strides=(2, 2), padding="SAME"), # 5
            layers.BatchNormalization(),
            layers.Activation(tf.nn.leaky_relu),
        ])

        self.discriminator = models.Sequential([
            layers.Flatten(),
            
            layers.Dense(256, activation=tf.nn.leaky_relu),
            layers.Dropout(drop_rate),
            
            layers.Dense(32, activation=tf.nn.leaky_relu),
            layers.Dropout(drop_rate),
            
            layers.Dense(10, activation=tf.nn.softmax),
        ])

    def call(self, inputs, training=False):
        x = self.features(inputs)
        x = self.discriminator(x)
        return x
