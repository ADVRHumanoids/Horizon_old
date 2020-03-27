from casadi import *

def create_bounds(dict, number_of_nodes):
    """Create a single list of bounds given the bounds on states, controls and optionally final time.
        Args:
            dict: {"x_min": x_min, "x_max": x_max, "u_min": u_min, "u_max": u_max}, optional: "tf_min": tf_min, "tf_max": tf_max
                For the first number_of_nodes-1 state and control bounds are set, then bounds on final time (tf_min/tf_max) are added if present,
                finally on last node only state bounds are added.
            number_of_nodes: total nodes of the problem

        Returns:
            vertcat(*v_min): list of all min bounds on state, controls and final time if present
            vertcat(*v_max): list of all max bounds on state, controls and final time if present
        """

    x_min = dict["x_min"]
    x_max = dict["x_max"]
    u_min = dict["u_min"]
    u_max = dict["u_max"]

    v_min = []
    v_max = []
    for k in range(number_of_nodes - 1):
        v_min += x_min
        v_max += x_max
        v_min += u_min
        v_max += u_max

    if "tf_min" in dict:
        if "tf_max" in dict:
            tf_min = dict["tf_min"]
            tf_max = dict["tf_max"]
            v_min.append(tf_min)
            v_max.append(tf_max)

    v_min += x_min
    v_max += x_max

    return vertcat(*v_min), vertcat(*v_max)


def create_init(dict, number_of_nodes):
    """Create a single list of initial conditions given the initial condition on states, controls and optionally final time.
            Args:
                dict: {"x_init": x_init, "u_init": u_init}, optional: "tf_init": tf_init
                    For the first number_of_nodes-1 state and control initial conditions are set, then initial condition on final time (tf_init) are added if present,
                    finally on last node only state initial conditions are added.
                number_of_nodes: total nodes of the problem

            Returns:
                vertcat(*v_init): list of all initial conditions on state, controls and final time if present
            """

    x_init = dict["x_init"]
    u_init = dict["u_init"]
    v_init = []
    for k in range(number_of_nodes - 1):
        v_init += x_init
        v_init += u_init
    if "tf_init" in dict:
        tf_init = dict["tf_init"]
        v_init.append(tf_init)
    v_init += x_init

    return vertcat(*v_init)

# concat_states_and_controls: creates a list concatenating (vertically) each variable concatenated variable contained in X
# and U at the same node adding the final state at the end:
# Example: X = [MX(vertcat(Q0, Qdot0)), MX(vertcat(Q1, Qdot1)), MX(vertcat(Q2, Qdot2)), ..., MX(vertcat(Qn, Qdotn))]'
#          U = [MX(vertcat(Qddot0, F0)), MX(vertcat(Qddot1, F1)), MX(vertcat(Qddot2, F2)), ..., MX(vertcat(Qddotn-1, Fn-1))]'
# then V = [vertcat(Q0, Qdot0, Qddot0, F0, Q1, Qdot1, Qddot1, F1, Q2, Qdot2, Qddot2, F2, ..., Qn, Qdotn]'
def concat_states_and_controls(dict):
    X = dict["X"]
    U = dict["U"]
    ns = np.size(U)
    V = []
    for k in range(ns):
        V.append(vertcat(X[k], U[k]))
    if "Tf" in dict:
        Tf = dict["Tf"]
        V.append(*Tf)
    V.append(X[ns])
    return vertcat(*V)



# concat: creates a list concatenating (vertically) each variable contained in V at the same node:
# Example: V = [Q, Qdot] then X = [MX(vertcat(Q0, Qdot0)), MX((vertcat(Q1, Qdot1)), MX(vertcat(Q2, Qdot2)), ...]'
def concat(V, s):
    X = []
    for k in range(s):
        x = []
        for j in V:
            x.append(j[k])
        X.append(vertcat(*x))
    return X

# create_state_and_control: creates state list X and control list U  from VX list and VU list using concat
def create_state_and_control(VX, VU):
    #MISSING CHECK OF SIZE OF VARIABLES CONTAINED IN VX!!!
    ns = np.size(VX[0])
    X = concat(VX, ns)

    # MISSING CHECK OF SIZE OF VARIABLES CONTAINED IN VU!!!
    ns = np.size(VU[0])
    U = concat(VU, ns)
    return X, U

# dynamic_model_with_floating_base: gets in input a q of size [n, 1] and a qdot of size [n-1, 1]
# (notice that the first 7 elements of q are postion and orientation with quaternion)
# and return the dynamic model x, xdot considering the integration of the quaterion for the floating base orientation
def dynamic_model_with_floating_base(q, qdot, qddot):
    # Model equations
    S = SX.zeros(3, 3)
    S[0, 1] = -q[5]
    S[0, 2] = q[4]
    S[1, 0] = q[5]
    S[1, 2] = -q[3]
    S[2, 0] = -q[4]
    S[2, 1] = q[3]

    # Quaternion Integration
    tmp1 = casadi.mtimes(0.5 * (q[6] * SX.eye(3) - S), qdot[3:6])
    tmp2 = -0.5 * casadi.mtimes(q[3:6].T, qdot[3:6])

    x = vertcat(q, qdot)
    xdot = vertcat(qdot[0:3], tmp1, tmp2, qdot[6:qdot.shape[0]], qddot)

    return x, xdot

# create_variable: return a SX_var of size [size, 1] and a MX_var of size [size, ns] where:
# if type is STATE then ns = number_of_nodes
# if type is CONTROL then ns = number_of_nodes-1
# if type is FINAL_STATE then ns = 1
# else ns = 0
#
# casadi_type can be 'SX' (default) or 'MX'
#
# output: SX_var = [SX_Q_0, SX_Q_1, ..., SX_Q_size] which represents the state variable for a single shooting node
#         MX_var = [MX(Q0), MX(Q1), ..., MX(Qnumber_of_nodes-1)] when type is CONTROL and casadi_type is 'MX', which are all the variables in the in all the nodes
def create_variable(name, size, number_of_nodes, type, casadi_type = 'SX'):
    SX_var = SX.sym('SX_'+name, size)
    opc_var = []

    ns = 0
    if type == "STATE":
        ns = number_of_nodes
    elif type == "CONTROL":
        ns = number_of_nodes-1
    elif type == "FINAL_STATE":
        ns = 1

    if casadi_type is 'MX':
        for i in range(ns):
            opc_var.append(MX.sym(name + str(i), SX_var.size1()))
    elif casadi_type is 'SX':
        for i in range(ns):
            opc_var.append(SX.sym(name + str(i), SX_var.size1()))
    else:
        raise Exception('casadi_type can be only SX or MX')

    return SX_var, opc_var

def cost_function(cost, from_node, to_node):
    """Apply a cost from an interval of nodes [from_node, to_node).
    Args:
        cost: a callable function which return the value of the cost for a given node k
        from_node: starting node (included)
        to_node: final node (excluded)

    Returns:
        J: list of values of the cost on the given nodes
    """
    J = []
    if type(cost(0)) is casadi.SX:
        J = SX([0])
    elif type(cost(0)) is casadi.MX:
        J = MX([0])
    else:
        raise Exception('Cost type can be only casadi.SX or casadi.MX!')

    for k in range(from_node, to_node):
        J += cost(k)
    return J

def constraint(constraint, from_node, to_node):
    """Apply a constraint from an interval of nodes [from_node, to_node).
    Args:
        constraint: a callable function which return the value of the constraint, min and max for a given node k
        from_node: starting node (included)
        to_node: final node (excluded)

    Returns:
        g: list of values of the constraint on the given nodes
        g_min: list of values of the lower bounds on the given nodes
        g_max: list of values of the upper bounds on the given nodes
    """

    g = []
    g_min = []
    g_max = []

    for k in range(from_node, to_node):
        gk, g_mink, g_maxk = constraint(k)
        g += gk
        g_min += g_mink
        g_max += g_maxk
    return g, g_min, g_max


class constraint_class:
    """Base class to implement constraints to be used in constraint(constraint, from_node, to_node).
        Attributes:
            gk: value of the constraint for the k node
            g_mink: value of the min bound for the k node
            g_maxk: value of the max bound for the k node
        """

    def __init__(self):
        self.gk = []
        self.g_mink = []
        self.g_maxk = []

    def __call__(self, k):
        """This method is automatically called when passed to constraint(constraint, from_node, to_node), here virtual_method is called.
         Args:
            k: node for which virtual_method is evaluated

        Returns:
            gk: value of the constraint for the k node
            g_mink: value of the min bound for the k node
            g_maxk: value of the max bound for the k node
        """
        self.virtual_method(k)
        return self.gk, self.g_mink, self.g_maxk

    def getConstraint(self):
        """Return the value of the constraint (virtual_method needs to be called first)
        #TODO: refactor to have getConstraint(k); virtual_method(k) is called inside
        Returns:
            gk: value of the constraint for the k node
            g_mink: value of the min bound for the k node
            g_maxk: value of the max bound for the k node
        """
        return self.gk, self.g_mink, self.g_maxk

    def virtual_method(self, k):
        """Method to implement in derived class, should provide values for gk, g_mink, g_maxk
        Args:
            k: node for which virtual_method is evaluated

        Raise:
            NotImplementedError() if not implmented in derived class
        """
        raise NotImplementedError()


class multiple_shooting(constraint_class):
    def __init__(self, dict, F_integrator):
        self.dict = dict
        self.F_integrator = F_integrator

        self.keys = []
        for key in self.dict:
            self.keys.append(key)

    def virtual_method(self, k):
        if 'time' in self.dict:  # time is optimized
            if np.size(self.dict['time']) == 1:  # only final time is optimized
                integrator_out = self.F_integrator(x0=self.dict['x0'][k], p=self.dict['p'][k],
                                                   time=self.dict['time'][0] / np.size(self.dict['p']))
            else:  # intermediate times are control variables
                integrator_out = self.F_integrator(x0=self.dict['x0'][k], p=self.dict['p'][k],
                                                   time=self.dict['time'][k])
        else:
            # time is not optimized
            integrator_out = self.F_integrator(x0=self.dict['x0'][k], p=self.dict['p'][k])

        self.gk = [integrator_out['xf'] - self.dict['x0'][k + 1]]
        self.g_mink = [0] * self.dict['x0'][k + 1].size1()
        self.g_maxk = [0] * self.dict['x0'][k + 1].size1()



class constraint_handler():
    def __init__(self):
        self.g = []
        self.g_min = []
        self.g_max = []

    def set_constraint(self, g, g_min, g_max):
        self.g += g
        self.g_min += g_min
        self.g_max += g_max

    def get_constraints(self):
        return vertcat(*self.g), self.g_min, self.g_max


def retrieve_solution(input, output_dict, solution):
    output_keys = []
    outputs = []
    ns = []
    for key in output_dict:
        output_keys.append(key)
        outputs += [vertcat(*output_dict[key])]
        ns.append(np.size(output_dict[key]))

    Retrieve = Function("Retrieve", [input], outputs, ['V'], output_keys)

    o = {}
    for i in range(len(output_keys)):
        tmp = Retrieve(V=solution)[output_keys[i]].full().flatten()
        nq = np.size(tmp)/ns[i]
        o[output_keys[i]] = tmp.reshape(ns[i], nq)

    return o





