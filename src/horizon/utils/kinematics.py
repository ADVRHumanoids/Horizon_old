from casadi import *
from horizon import *


class kinematics:
    def __init__(self, kindyn, Q):
        self.kindyn = kindyn
        self.Q = Q

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
