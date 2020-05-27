import foolbox as fb
import tensorflow as tf

from adversarial.attacks import AttackMethod


class LinfDeepFool(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(LinfDeepFool, self).__init__(tfmodel, epsilons)
        self.name = "LinfDeepFool"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.LinfDeepFoolAttack()


class L2DeepFool(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(L2DeepFool, self).__init__(tfmodel, epsilons)
        self.name = "L2DeepFool"

    def setup(self):
        super().setup()
        self.attack = fb.attacks.L2DeepFoolAttack()


AttackMethod.register(LinfDeepFool)
AttackMethod.register(L2DeepFool)
