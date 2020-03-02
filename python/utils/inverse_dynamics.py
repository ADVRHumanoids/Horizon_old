from casadi import *
from horizon import *


class inverse_dynamics:
    def __init__(self, Q, Qdot, Qddot, ID, dict, kindyn):
        self.Q = Q
        self.Qdot = Qdot
        self.Qddot = Qddot
        self.ID = ID
        self.dict = dict
        self.kindyn = kindyn

    def compute(self, k):

        JtF_k = MX([0])

        for key in self.dict:
            Jac = Function.deserialize(self.kindyn.jacobian(key))
            Jac_k = Jac(q=self.Q[k])['J']
            Force_k = self.dict[key][k]
            JtF = mtimes(Jac_k.T, vertcat(Force_k, MX.zeros(3, 1)))
            JtF_k = JtF_k + JtF

        return (self.ID(q=self.Q[k], v=self.Qdot[k], a=self.Qddot[k])['tau'] - JtF_k)


    def compute_nodes(self, from_node, to_node):

        tau = []

        for k in range(from_node, to_node):
            tau.append(self.compute(k))

        return vertcat(*tau)

class inverse_dynamicsSX:
    def __init__(self, Q, Qdot, Qddot, ID, dict, kindyn):
        self.Q = Q
        self.Qdot = Qdot
        self.Qddot = Qddot
        self.ID = ID
        self.dict = dict
        self.kindyn = kindyn

    def compute(self, k):

        JtF_k = SX([0])

        for key in self.dict:
            Jac = Function.deserialize(self.kindyn.jacobian(key))
            Jac_k = Jac(q=self.Q[k])['J']
            Force_k = self.dict[key][k]
            JtF = mtimes(Jac_k.T, vertcat(Force_k, SX.zeros(3, 1)))
            JtF_k = JtF_k + JtF

        return (self.ID(q=self.Q[k], v=self.Qdot[k], a=self.Qddot[k])['tau'] - JtF_k)


    def compute_nodes(self, from_node, to_node):

        tau = []

        for k in range(from_node, to_node):
            tau.append(self.compute(k))

        return vertcat(*tau)