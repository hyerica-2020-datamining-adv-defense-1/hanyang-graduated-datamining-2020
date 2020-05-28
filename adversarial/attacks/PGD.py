import foolbox as fb
import tensorflow as tf

from adversarial.attacks import AttackMethod


class PGD(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(PGD, self).__init__(tfmodel, epsilons)
        self.name = "PGD"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.PGD()


class LinfPGD(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(LinfPGD, self).__init__(tfmodel, epsilons)
        self.name = "LinfPGD"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.LinfPGD()


class L2PGD(AttackMethod):

    def __init__(self, tfmodel, epsilons):
        super(L2PGD, self).__init__(tfmodel, epsilons)
        self.name = "L2PGD"
        
    def setup(self):
        super().setup()
        self.attack = fb.attacks.L2PGD()


AttackMethod.register(PGD)
AttackMethod.register(LinfPGD)
AttackMethod.register(L2PGD)
