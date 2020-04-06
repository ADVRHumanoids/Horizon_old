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
        # FK_vel = Function.deserialize(self.kindyn.frameVelocity(link_name))
        # FK_acc = Function.deserialize(self.kindyn.frameAcceleration(link_name))
        Jac = Function.deserialize(self.kindyn.jacobian(link_name))

        link_fk = []
        if fk_type is 'ee_pos':
            for k in range(from_node, to_node):
                link_fk.append(FK(q=self.Q[k])['ee_pos'])
        elif fk_type is 'ee_rot':
            for k in range(from_node, to_node):
                link_fk.append(FK(q=self.Q[k])['ee_rot'])
        elif fk_type is 'ee_vel_linear':
            for k in range(from_node, to_node):
                Jac_k = Jac(q=self.Q[k])['J']
                twist_k = mtimes(Jac_k, self.Qdot[k])
                link_fk.append(twist_k[0:3])
                # link_fk.append(FK_vel(q=self.Q[k], qdot=self.Qdot[k])['ee_vel_linear'])
        elif fk_type is 'ee_vel_angular':
            for k in range(from_node, to_node):
                Jac_k = Jac(q=self.Q[k])['J']
                twist_k = mtimes(Jac_k, self.Qdot[k])
                link_fk.append(twist_k[3:6])
                # link_fk.append(FK_vel(q=self.Q[k], qdot=self.Qdot[k])['ee_vel_angular'])
        elif fk_type is 'ee_acc_linear':
            for k in range(from_node, to_node):
                Jac_k = Jac(q=self.Q[k])['J']
                acc_k = mtimes(Jac_k, self.Qddot[k])
                link_fk.append(acc_k[0:3])
                # link_fk.append(FK_acc(q=self.Q[k], qdot=self.Qdot[k], qddot=self.Qddot[k])['ee_acc_linear'])
        elif fk_type is 'ee_acc_angular':
            for k in range(from_node, to_node):
                Jac_k = Jac(q=self.Q[k])['J']
                acc_k = mtimes(Jac_k, self.Qddot[k])
                link_fk.append(acc_k[3:6])
                # link_fk.append(FK_acc(q=self.Q[k], qdot=self.Qdot[k], qddot=self.Qddot[k])['ee_acc_linear'])
        else:
            raise NotImplementedError()

        return vertcat(*link_fk)

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
