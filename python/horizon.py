from casadi import *


# create_bounds: creates a list of bounds, along the number of knots, for the variables given the list of bounds
# on states x and controls u
def create_bounds(x_min, x_max, u_min, u_max, number_of_nodes):
    v_min = []
    v_max = []
    for k in range(number_of_nodes):
        v_min += x_min
        v_max += x_max
        v_min += u_min
        v_max += u_max
    v_min += x_min
    v_max += x_max

    return v_min, v_max



# concat_states_and_controls: creates a list concatenating (vertically) each variable concatenated variable contained in X
# and U at the same node adding the final state at the end:
# Example: X = [vertcat(Q0, Qdot0), vertcat(Q1, Qdot1), vertcat(Q2, Qdot2), ..., vertcat(Qn, Qdotn)]'
#          U = [vertcat(Qddot0, F0), vertcat(Qddot1, F1), vertcat(Qddot2, F2), ..., vertcat(Qddotn-1, Fn-1)]'
# then V = [vertcat(Q0, Qdot0, Qddot0, F0), vertcat(Q1, Qdot1, Qddot1, F1), vertcat(Q2, Qdot2, Qddot2, F2), ..., vertcat(Qn, Qdotn)]'
def concat_states_and_controls(X, U):
    ns = np.size(U)
    V = []
    for k in range(ns):
        V.append(vertcat(X[k], U[k]))
    V.append(X[ns])
    return V

# concat: creates a list concatenating (vertically) each variable contained in V at the same node:
# Example: V = [Q, Qdot] then X = [vertcat(Q0, Qdot0), vertcat(Q1, Qdot1), vertcat(Q2, Qdot2), ...]'
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
# if type is STATE then ns = number_of_nodes + 1
# if type is CONTRON then ns = number_of_nodes
# if type is FINAL_STATE then ns = 1
# else ns = 0
def create_variable(name, size, number_of_nodes, type):
    SX_var = SX.sym('SX_'+name, size)
    MX_var = []

    ns = 0
    if type == "STATE":
        ns = number_of_nodes + 1
    elif type == "CONTROL":
        ns = number_of_nodes
    elif type == "FINAL_STATE":
        ns = 1

    for i in range(ns):
        MX_var.append(MX.sym(name + str(i), SX_var.size1()))

    return SX_var, MX_var

# cost_function return the value of cost (functor) computed from from_node to to_node
def cost_function(cost, from_node, to_node):
    J = MX([0])
    for k in range(from_node, to_node):
        J += cost(k)
    return J


def constraint(constraint, from_node, to_node):
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
    def __init__(self):
        self.gk = []
        self.g_mink = []
        self.g_maxk = []

    def __call__(self, k):
        self.virtual_method(k)
        return self.gk, self.g_mink, self.g_maxk

    def virtual_method(self, k):
        raise NotImplementedError()










