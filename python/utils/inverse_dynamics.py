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

        JtF_k = []
        zeros = []
        if type(self.Q[0]) is casadi.SX:
            JtF_k = SX([0])
            zeros =  SX.zeros(3, 1)
        elif type(self.Q[0]) is casadi.MX:
            JtF_k = MX([0])
            zeros = MX.zeros(3, 1)
        else:
            raise Exception('Input type can be only casadi.SX or casadi.MX!')

        for key in self.dict:
            Jac = Function.deserialize(self.kindyn.jacobian(key, self.kindyn.LOCAL))
            Jac_k = Jac(q=self.Q[k])['J']
            Force_k = self.dict[key][k]
            JtF = mtimes(Jac_k.T, vertcat(Force_k, zeros))
            JtF_k = JtF_k + JtF

        return (self.ID(q=self.Q[k], v=self.Qdot[k], a=self.Qddot[k])['tau'] - JtF_k)


    def compute_nodes(self, from_node, to_node):

        tau = []

        for k in range(from_node, to_node):
            tau.append(self.compute(k))

        return vertcat(*tau)
