from Horizon.horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn

class position(constraint_class):
    """
    Constraint the position of a given frame to remain inside a bounding box
    """
    def __init__(self, FKlink, Q, min, max):
        """
        Constructor
        Args:
            FKlink: Forward kinematics of frame to constraint
            Q: state varibales
            min: lower bound list [xmin, ymin, zmin]
            max: upper bound list [xmax, ymax, zmax]
        """
        self.FKlink = FKlink
        self.Q = Q
        self.min = min
        self.max= max

    def virtual_method(self, k):
        """
        Compute constraint at k
        Args:
            k: node
        """
        CLink_pos = self.FKlink(q=self.Q[k])['ee_pos']
        self.gk = [CLink_pos]
        self.g_mink = np.array(self.min).tolist()
        self.g_maxk = np.array(self.max).tolist()
