from casadi import *
from Horizon.horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn


class inverse_dynamics:
    """
    Class which computes inverse dynamics:
    given generalized position, velocities and acceleretaions returns generalized torques

    TODO: Extend to [force torques] vector!
    """

    def __init__(self, Q, Qdot, Qddot, ID, dict, kindyn, force_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL):
        """
        Constructor
        Args:
            Q: joint positions
            Qdot: joint velocities
            Qddot: joint accelerations
            ID: Function.deserialize(kindyn.rnea()) TODO: remove, this can be taken from kindyn
            dict: dictionary containing a map between frames and force variables e.g. {'lsole': F1}
            kindyn: casadi_kin_dyn object
            force_reference_frame: this is the frame which is used to compute the Jacobian during the ID computation:
                LOCAL (default)
                WORLD
                LOCAL_WORLD_ALIGNED
        """
        self.Q = Q
        self.Qdot = Qdot
        self.Qddot = Qddot
        self.ID = ID
        self.dict = dict
        self.kindyn = kindyn
        self.force_ref_frame = force_reference_frame

    def compute(self, k):
        """
        Compute torques at k node
        Args:
            k: node

        Returns:
            tauk = (self.ID(q=self.Q[k], v=self.Qdot[k], a=self.Qddot[k])['tau'] - JtF_k)
        """

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
            Jac = Function.deserialize(self.kindyn.jacobian(key, self.force_ref_frame))
            Jac_k = Jac(q=self.Q[k])['J']
            Force_k = self.dict[key][k]
            JtF = mtimes(Jac_k.T, vertcat(Force_k, zeros))
            JtF_k = JtF_k + JtF

        return (self.ID(q=self.Q[k], v=self.Qdot[k], a=self.Qddot[k])['tau'] - JtF_k)


    def compute_nodes(self, from_node, to_node):
        """
        Compute torques from_node to_node
        Args:
            from_node: starting node
            to_node: end node

        Returns:
            tau = list of torques one for each node
        """

        tau = []

        for k in range(from_node, to_node):
            tau.append(self.compute(k))

        return vertcat(*tau)
