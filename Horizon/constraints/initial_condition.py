from Horizon.horizon import *

class initial_condition(constraint_class):
    """
    Given a single node variable impose a state
    TODO: remove, better the state_conditio, is more general
    """
    def __init__(self, X0, x_init):
        """
        Constructor
        Args:
            X0: Variable at a single node
            x_init: reference
        """
        self.X0 = X0
        self.x_init = x_init

    def virtual_method(self, k):
        """
        Compute constraint for a node k
        Args:
            k: node
        """
        self.gk = [self.X0]
        self.g_mink = self.x_init
        self.g_maxk = self.x_init

class state_condition(constraint_class):
    """
    Impose a certaint condition x_d on a given variable X for a certain number of nodes
    """
    def __init__(self, X, x_d):
        """
        Constructor
        Args:
            X: Variable
            xd: reference
        """
        self.X = X
        self.x_d = x_d

    def virtual_method(self, k):
        """
        Compute constraint for a node k
        Args:
            k: node
        """
        self.gk = [self.X[k]]
        self.g_mink = self.x_d
        self.g_maxk = self.x_d
