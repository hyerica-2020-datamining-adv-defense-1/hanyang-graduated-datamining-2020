import foolbox as fb
import tensorflow as tf

from adversarial.attacks import AttackMethod


class CarliniWagner(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(CarliniWagner, self).__init__(tfmodel, epsilons)
        self.name = "C&W"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.L2CarliniWagnerAttack()


#AttackMethod.register(CarliniWagner)
