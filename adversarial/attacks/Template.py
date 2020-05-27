import foolbox as fb
import tensorflow as tf

from adversarial.attacks import AttackMethod


# class Template(AttackMethod):

#     def __init__(self, tfmodel, epsilons):
#         super(Template, self).__init__(tfmodel, epsilons)
#         self.name = "TEMPLATE"
        
#     def setup(self):
#         super().setup()
#         self.attack = fb.attacks.FGSM()


# AttackMethod.register(Template)
