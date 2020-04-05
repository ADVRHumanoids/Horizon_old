from casadi import *
from horizon import *


class kinematics:
    def __init__(self, kindyn, Q, Qdot, Qddot):
        self.kindyn = kindyn
        self.Q = Q
        self.Qdot = Qdot
        self.Qddot = Qddot

    def computeFK(self, link_name, fk_type, from_node, to_node):
        FK = Function.deserialize(self.kindyn.fk(link_name))

        link_pos = []
        if fk_type is 'ee_pos':
            for k in range(from_node, to_node):
                link_pos.append(FK(q=self.Q[k])['ee_pos'])
        elif fk_type is 'ee_rot':
            for k in range(from_node, to_node):
                link_pos.append(FK(q=self.Q[k])['ee_rot'])
        else:
            raise NotImplementedError()

        return vertcat(*link_pos)

    def computeCoM(self, fk_type, from_node, to_node):
        FK = Function.deserialize(self.kindyn.centerOfMass())

        CoM = []

        if fk_type is 'com':
            for k in range(from_node, to_node):
                CoM.append(FK(q=self.Q[k])['com'])
        elif fk_type is 'vcom':
            for k in range(from_node, to_node):
                CoM.append(FK(q=self.Q[k], v=self.Qdot[k])['vcom'])
        elif fk_type is 'acom':
            for k in range(from_node, to_node):
                CoM.append(FK(q=self.Q[k], v=self.Qdot[k], a=self.Qddot[k])['acom'])
        else:
            raise NotImplementedError()

        return vertcat(*CoM)
