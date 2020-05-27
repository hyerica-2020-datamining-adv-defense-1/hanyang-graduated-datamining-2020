import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, applications


class BaseModel(models.Model):

    def __init__(self, base_model):
        super(BaseModel, self).__init__()

        self.base_model = base_model
        self.top_layer = models.Sequential([
            layers.Dense(10),
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
        x = self.base_model(inputs, training=training)
        x = layers.Flatten()(x)
        outputs = self.top_layer(x, training=training)

        return outputs


class MobileNetV2(BaseModel):

    def __init__(self):
        super(MobileNetV2, self).__init__(
            applications.MobileNetV2(input_shape=[160, 160, 3], include_top=False, weights=None)
        )

TargetModel = MobileNetV2


class VGG16(BaseModel):

    def __init__(self):
        super(VGG16, self).__init__(
            applications.VGG16(input_shape=[160, 160, 3], include_top=False, weights="imagenet")
        )


class VGG19(BaseModel):

    def __init__(self):
        super(VGG19, self).__init__(
            applications.VGG19(input_shape=[160, 160, 3], include_top=False, weieghts="imagenet")
        )


class MobileNet(BaseModel):

    def __init__(self):
        super(MobileNet, self).__init__(
            applications.MobileNet(input_shape=(160, 160, 3), include_top=False, weights="imagenet")
        )


class ResNet50(BaseModel):

    def __init__(self):
        super(ResNet50, self).__init__(
            applications.ResNet50(input_shape=(160, 160, 3), include_top=False, weights="imagenet")
        )


class ResNet50V2(BaseModel):

    def __init__(self):
        super(ResNet50V2, self).__init__(
            applications.ResNet50V2(input_shape=(160, 160, 3), include_top=False, weights="imagenet")
        )

