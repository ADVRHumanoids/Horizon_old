from horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn

class contact(constraint_class):
    def __init__(self, FKlink, Q, qinit):
        self.FKlink = FKlink
        self.Q = Q
        self.qinit = qinit

    def virtual_method(self, k):
        CLink_pos_init = self.FKlink(q=self.qinit)['ee_pos']
        CLink_pos = self.FKlink(q=self.Q[k])['ee_pos']
        self.gk = [CLink_pos - CLink_pos_init]
        self.g_mink = np.array([0., 0., 0.]).tolist()
        self.g_maxk = np.array([0., 0., 0.]).tolist()