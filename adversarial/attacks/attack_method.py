import foolbox as fb
import tensorflow as tf
from tensorflow.keras import layers, models


class AttackMethod():

    attack_methods = []

    def __init__(self, tfmodel, epsilons):
        tfmodel.trainable = False
        self.fbmodel = fb.TensorFlowModel(tfmodel, bounds=[0.0, 1.0], preprocessing=dict())
        self.epsilons = epsilons
        self.attack = None
        self.is_setup_called = False
        self.name = None

    def setup(self):
        self.is_setup_called = True

    def call(self, images, labels):
        assert self.is_setup_called == True, "You should call setup() first before call()."
        assert self.attack is not None, "AttackMethod.attack attribute must not be None."

        images = tf.convert_to_tensor(images)
        labels = tf.convert_to_tensor(labels)

        adv, clipped, success = self.attack(self.fbmodel, images, labels, epsilons=self.epsilons)
        return adv, clipped, success

    def __call__(self, images, labels):
        return self.call(images, labels)

    @classmethod
    def register(cls, attack):
        cls.attack_methods.append(attack)

    @classmethod
    def attack(cls, tfmodel, images, labels, epsilons):
        results = dict()

        for attack_method_cls in cls.attack_methods:
            attack_method = attack_method_cls(tfmodel, epsilons)
            attack_method.setup()

            adv, clipped, success = attack_method(images, labels)
            results[attack_method.name] = (adv, clipped, success)

        return results


def attack(tfmodel, images, labels, epsilons):
    return AttackMethod.attack(tfmodel, images, labels, epsilons)
