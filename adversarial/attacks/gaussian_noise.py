import foolbox as fb
import tensorflow as tf

from adversarial.attacks import AttackMethod


class GaussianBlur(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(GaussianBlur, self).__init__(tfmodel, epsilons)
        self.name = "GaussianBlur"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.GaussianBlurAttack()


class AdditiveGaussianNoise(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(AdditiveGaussianNoise, self).__init__(tfmodel, epsilons)
        self.name = "L2AdditiveGaussianNoiseAttack"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.L2AdditiveGaussianNoiseAttack()


class RepeatedAdditiveGaussianNoise(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(RepeatedAdditiveGaussianNoise, self).__init__(tfmodel, epsilons)
        self.name = "RepeatedAdditiveGaussianNoise"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.L2RepeatedAdditiveGaussianNoiseAttack()
        

AttackMethod.register(GaussianBlur)
AttackMethod.register(AdditiveGaussianNoise)
AttackMethod.register(RepeatedAdditiveGaussianNoise)
