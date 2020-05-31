import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from adversarial.defences.feature_denoise import FeatureDenoisingBlock


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
            applications.MobileNetV2(input_shape=[160, 160, 3], include_top=False, weights="imagenet")
        )

# TargetModel = MobileNetV2
class TargetModel(models.Model):

    def __init__(self):
        super(TargetModel, self).__init__()

        self.base_model = MobileNetV2()
        self.add_block_num = []
        
        feature_denoising_blocks = []

        for layer in self.base_model.base_model.layers:
            out_channel = layer.output_shape[-1]
            if layer.name.endswith("_add"):
                feature_denoising_blocks.append(FeatureDenoisingBlock(out_channel))
                self.add_block_num.append(int(layer.name.split("_")[-2]))
                
        self.feature_denoising_blocks = feature_denoising_blocks

    def load_custom_weights_for_mobilenet(self, path):
        self.base_model.build(input_shape=(None, 160, 160, 3))
        with open(path, "rb") as f:
            weights = pickle.load(f)
            self.base_model.set_weights(weights)

    def load_custom_weights(self, path):
        with open(path, "rb") as f:
            weights = pickle.load(f)
            self.set_weights(weights)

    def save_custom_weights(self, path):
        with open(path, "wb") as f:
            weights = self.get_weights()
            pickle.dump(weights, f)

    def call(self, inputs, training=False):
        n = tf.shape(inputs)[0]
        
        x = inputs
        cnt = 0

        residual_input = None

        feature_maps = []

        for layer in self.base_model.base_model.layers:
            if residual_input is None and cnt < len(self.add_block_num) and layer.name.startswith(f"block_{self.add_block_num[cnt]}"):
                residual_input = x

            if layer.name.endswith("_add"):
                x = layer([residual_input, x], training=training)
                feature_maps.append(x)
                x = self.feature_denoising_blocks[cnt](x, training=training)
                residual_input = None
                cnt += 1
            else:
                x = layer(x, training=training)

        # x = self.base_model.base_model(x)

        x = tf.reshape(x, shape=(n, -1))
        outputs = self.base_model.top_layer(x, training=training)

        return outputs , feature_maps


class VGG16(BaseModel):

    def __init__(self):
        super(VGG16, self).__init__(
            applications.VGG16(input_shape=[160, 160, 3], include_top=False, weights="imagenet")
        )


class VGG19(BaseModel):

    def __init__(self):
        super(VGG19, self).__init__(
            applications.VGG19(input_shape=[160, 160, 3], include_top=False, weights="imagenet")
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

