from casadi import *
from Horizon.horizon import *

class kinematics:
    """
    The kinematics class provides easy accessor to computation of FK quantities such as Cartesian poses and veocities

    """
    def __init__(self, kindyn, Q, Qdot, Qddot):
        """
        Constructor
        Args:
            kindyn: casadi_kin_dyn object
            Q: state positions
            Qdot: state velocities
            Qddot: control accelerations
        """
        self.kindyn = kindyn
        self.Q = Q
        self.Qdot = Qdot
        self.Qddot = Qddot

    def computeKineticEnergy(self, from_node, to_node):
        """
        Computes kinetic energy of the system:

            0.5*m*v^2

        for a given interval of nodes.

        Args:
            from_node: starting node
            to_node: end node

        Returns:
            vertcat(*DT) of kinetic energies

        """
        KineticEnergy = Function.deserialize(self.kindyn.kineticEnergy())
        DT = []
        for k in range(from_node, to_node):
            DT.append(KineticEnergy(q=self.Q[k], v=self.Qdot[k])['DT'])
        return vertcat(*DT)

    def computePotentialEnergy(self, from_node, to_node):
        """
        Computes kinetic energy of the system:

            m*g*h

        for a given interval of nodes.

        Args:
            from_node: starting node
            to_node: end node

        Returns:
            vertcat(*DT) of potential energies

        """
        PotentialEnergy = Function.deserialize(self.kindyn.potentialEnergy())
        DU = []
        for k in range(from_node, to_node):
            DU.append(PotentialEnergy(q=self.Q[k])['DU'])
        return vertcat(*DU)

    def computeDiffFK(self, link_name, fk_type, ref, from_node, to_node):
        """
        Computes differential forward kinematics quantities: velocities and Jacobians ///TODO: Add accelerations
        for a given number of nodes.
        Args:
            link_name: frame wrt computes quantities
            fk_type:
                'ee_vel_linear' for linear velocity
                'ee_vel_angular' for angular velocity
                'ee_jacobian' for jacobian
            ref: this is the frame which is used to compute the Jacobian during the ID computation:
                LOCAL
                WORLD
                LOCAL_WORLD_ALIGNED
            from_node: start node
            to_node: end node

        Returns:

        """
        FK_vel = Function.deserialize(self.kindyn.frameVelocity(link_name, ref))
        # FK_acc = Function.deserialize(self.kindyn.frameAcceleration(link_name))
        Jac = Function.deserialize(self.kindyn.jacobian(link_name, ref))

        link_fk = []
        if fk_type is 'ee_vel_linear':
            for k in range(from_node, to_node):
                link_fk.append(FK_vel(q=self.Q[k], qdot=self.Qdot[k])['ee_vel_linear'])
        elif fk_type is 'ee_vel_angular':
            for k in range(from_node, to_node):
                link_fk.append(FK_vel(q=self.Q[k], qdot=self.Qdot[k])['ee_vel_angular'])
        # elif fk_type is 'ee_acc_linear':
        #    for k in range(from_node, to_node):
        #        link_fk.append(FK_acc(q=self.Q[k], qdot=self.Qdot[k], qddot=self.Qddot[k])['ee_acc_linear'])
        # elif fk_type is 'ee_acc_angular':
        #    for k in range(from_node, to_node):
        #        link_fk.append(FK_acc(q=self.Q[k], qdot=self.Qdot[k], qddot=self.Qddot[k])['ee_acc_angular'])
        elif fk_type is 'ee_jacobian':
            for k in range(from_node, to_node):
                link_fk.append(Jac(q=self.Q[k])['J'])
        else:
            raise NotImplementedError()

        return vertcat(*link_fk)




    def computeFK(self, link_name, fk_type, from_node, to_node):
        FK = Function.deserialize(self.kindyn.fk(link_name))

        link_fk = []
        if fk_type is 'ee_pos':
            for k in range(from_node, to_node):
                link_fk.append(FK(q=self.Q[k])['ee_pos'])
        elif fk_type is 'ee_rot':
            for k in range(from_node, to_node):
                link_fk.append(FK(q=self.Q[k])['ee_rot'])
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
