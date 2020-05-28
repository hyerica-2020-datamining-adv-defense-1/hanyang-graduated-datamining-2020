import tensorflow as tf
from tensorflow.keras import layers, models


class FeatureDenoisingBlock(models.Model):

    def __init__(self, in_channels):
        super(FeatureDenoisingBlock, self).__init__()

        self.embedding1 = layers.Conv2D(in_channels, (1, 1), strides=(1, 1), padding="SAME")
        self.embedding2 = layers.Conv2D(in_channels, (1, 1), strides=(1, 1), padding="SAME")
        self.embedding3 = layers.Conv2D(in_channels, (1, 1), strides=(1, 1), padding="SAME")

    def call(self, inputs, training=False):
        n, h, w, c = tf.shape(inputs)

        theta, phi = self._embedding(inputs, training)
        gaussian_channel = self._compute_gaussian_channel(theta, phi)
        softmax = tf.nn.softmax(gaussian_channel, axis=-1)
        denoised = self._denoising(inputs, softmax, training)

        return inputs + denoised

    def _embedding(self, inputs, training):
        n, h, w, c = tf.shape(inputs)

        theta = self.embedding1(inputs, training=training)
        phi = self.embedding2(inputs, training=training)

        theta = tf.reshape(theta, shape=(n, h*w, c))
        phi = tf.reshape(phi, shape=(n, h*w, c))
        phi = tf.transpose(phi, (0, 2, 1))

        return theta, phi

    def _compute_gaussian_channel(self, theta, phi):
        n = tf.shape(theta)[0]

        log_gaussian_channels = []

        for i in range(n):
            log_gaussian_channels.append(tf.matmul(theta[i], phi[i]))

        log_gaussian_channel = tf.stack(log_gaussian_channels, axis=0)
        gaussian_channel = tf.math.exp(log_gaussian_channel)

        return gaussian_channel

    def _denoising(self, inputs, softmax, training):
        n, h, w, c = tf.shape(inputs)
        
        inputs_reshaped = tf.reshape(inputs, shape=(n, h*w, c))

        denoised = []

        for i in range(n):
            denoised.append(tf.matmul(softmax[i], inputs_reshaped[i]))

        denoised = tf.stack(denoised, axis=0)
        denoised = tf.reshape(denoised, shape=(n, h, w, c))
        denoised = self.embedding3(denoised, training=training)

        return denoised
