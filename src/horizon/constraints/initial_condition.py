from horizon import *

class initial_condition(constraint_class):
    def __init__(self, X0, x_init):
        self.X0 = X0
        self.x_init = x_init

    def virtual_method(self, k):
        self.gk = [self.X0]
        self.g_mink = self.x_init
        self.g_maxk = self.x_init

