# import foolbox as fb
# import tensorflow as tf

# from adversarial.attacks import AttackMethod


# class L2AdditiveUniformNoise(AttackMethod):

#     def __init__(self, tfmodel, epsilons):
#         super(L2AdditiveUniformNoise, self).__init__(tfmodel, epsilons)
#         self.name = "L2AdditiveUniformNoise"
        
#     def setup(self):
#         super().setup()
#         self.attack = fb.attacks.L2AdditiveUniformNoiseAttack()


# class LinfAdditiveUniformNoise(AttackMethod):

#     def __init__(self, tfmodel, epsilons):
#         super(LinfAdditiveUniformNoise, self).__init__(tfmodel, epsilons)
#         self.name = "LinfAdditiveUniformNoise"
        
#     def setup(self):
#         super().setup()
#         self.attack = fb.attacks.LinfAdditiveUniformNoiseAttack()


# class L2RepeatedAdditiveUniformNoise(AttackMethod):

#     def __init__(self, tfmodel, epsilons):
#         super(L2RepeatedAdditiveUniformNoise, self).__init__(tfmodel, epsilons)
#         self.name = "L2RepeatedAdditiveUniformNoise"
        
#     def setup(self):
#         super().setup()
#         self.attack = fb.attacks.L2RepeatedAdditiveUniformNoiseAttack()


# class LinfRepeatedAdditiveUniformNoise(AttackMethod):

#     def __init__(self, tfmodel, epsilons):
#         super(LinfRepeatedAdditiveUniformNoise, self).__init__(tfmodel, epsilons)
#         self.name = "LinfRepeatedAdditiveUniformNoise"
        
#     def setup(self):
#         super().setup()
#         self.attack = fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack()



# AttackMethod.register(L2AdditiveUniformNoise)
# AttackMethod.register(LinfAdditiveUniformNoise)
# AttackMethod.register(L2RepeatedAdditiveUniformNoise)
# AttackMethod.register(LinfRepeatedAdditiveUniformNoise)
