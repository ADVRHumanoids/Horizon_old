from horizon import *
from utils.inverse_dynamics import *

class torque_lims(constraint_class):
    def __init__(self, id, tau_min, tau_max):
        self.id = id
        self.tau_min = tau_min
        self.tau_max = tau_max

    def virtual_method(self, k):
        self.gk = [self.id.compute(k)]
        self.g_mink = self.tau_min
        self.g_maxk = self.tau_max

class torque_lims_fb(constraint_class):
    def __init__(self, id):
        self.id = id

    def virtual_method(self, k):
        self.gk = [self.id.compute(k)[0:6]]
        self.g_mink = np.zeros(6).tolist()
        self.g_maxk = np.zeros(6).tolist()