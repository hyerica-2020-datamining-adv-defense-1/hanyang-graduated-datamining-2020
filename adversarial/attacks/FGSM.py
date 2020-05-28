import foolbox as fb
import tensorflow as tf

from adversarial.attacks import AttackMethod


class FGSM(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(FGSM, self).__init__(tfmodel, epsilons)
        self.name = "FGSM"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.FGSM()


AttackMethod.register(FGSM)
