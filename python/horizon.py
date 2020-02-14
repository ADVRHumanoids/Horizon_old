from casadi import *



# create_bounds: creates a list of bounds, along the number of knots, for the variables given the list of bounds
# on states x and controls u
#
# Output: v_min = [v_min1, v_min2, ..., v_minn]
def create_bounds(x_min, x_max, u_min, u_max, number_of_nodes):
    v_min = []
    v_max = []
    for k in range(number_of_nodes-1):
        v_min += x_min
        v_max += x_max
        v_min += u_min
        v_max += u_max
    v_min += x_min
    v_max += x_max

    return vertcat(*v_min), vertcat(*v_max)

def create_init(x_init, u_init, number_of_nodes):
    v_init = []
    for k in range(number_of_nodes - 1):
        v_init += x_init
        v_init += u_init
    v_init += x_init

    return vertcat(*v_init)



# concat_states_and_controls: creates a list concatenating (vertically) each variable concatenated variable contained in X
# and U at the same node adding the final state at the end:
# Example: X = [MX(vertcat(Q0, Qdot0)), MX(vertcat(Q1, Qdot1)), MX(vertcat(Q2, Qdot2)), ..., MX(vertcat(Qn, Qdotn))]'
#          U = [MX(vertcat(Qddot0, F0)), MX(vertcat(Qddot1, F1)), MX(vertcat(Qddot2, F2)), ..., MX(vertcat(Qddotn-1, Fn-1))]'
# then V = [vertcat(Q0, Qdot0, Qddot0, F0, Q1, Qdot1, Qddot1, F1, Q2, Qdot2, Qddot2, F2, ..., Qn, Qdotn]'
def concat_states_and_controls(X, U):
    ns = np.size(U)
    V = []
    for k in range(ns):
        V.append(vertcat(X[k], U[k]))
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
# output: SX_var = [SX_Q_0, SX_Q_1, ..., SX_Q_size]
#         MX_var = [MX(Q0), MX(Q1), ..., MX(Qnumber_of_nodes-1)] NOTE: if type is STATE
def create_variable(name, size, number_of_nodes, type):
    SX_var = SX.sym('SX_'+name, size)
    MX_var = []

    ns = 0
    if type == "STATE":
        ns = number_of_nodes
    elif type == "CONTROL":
        ns = number_of_nodes-1
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

class multiple_shooting(constraint_class):
    def __init__(self, X, Qddot, F_integrator):
        self.X = X
        self.Qddot = Qddot
        self.F_integrator = F_integrator

    def virtual_method(self, k):
        integrator_out = self.F_integrator(x0=self.X[k], p=self.Qddot[k])
        self.gk = [integrator_out['xf'] - self.X[k + 1]]
        self.g_mink = [0] * self.X[k + 1].size1()
        self.g_maxk = [0] * self.X[k + 1].size1()

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





