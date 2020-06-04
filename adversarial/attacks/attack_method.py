import foolbox as fb
import tensorflow as tf
import numpy as np
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

        if type(images) == np.ndarray:
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        if type(labels) == np.ndarray:
            labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        adv, clipped, success = self.attack(self.fbmodel, images, labels, epsilons=self.epsilons)
        return adv, clipped, success

    def __call__(self, images, labels):
        return self.call(images, labels)

    @classmethod
    def register(cls, attack):
        cls.attack_methods.append(attack)

    @classmethod
    def random_attack(cls, tfmodel, images, labels, epsilons):
        attack_method_cls = np.random.choice(cls.attack_methods)
        epsilon = np.random.choice(epsilons)

        attack_method = attack_method_cls(tfmodel, epsilon)
        attack_method.setup()
            
        # print(attack_method.name)

        adv, clipped, success = attack_method(images, labels)
        results = (adv, clipped, success)

        return results

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

def random_attack(tfmodel, images, labels, epsilons):
    return AttackMethod.random_attack(tfmodel, images, labels, epsilons)
