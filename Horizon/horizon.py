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

def concat_states_and_controls(dict):
    """
    Creates a single list of states and controls given the list of states and the list of controls
    Args:
        dict: a dictionary containing
            X: list of states
            U: list of controls
    Returns:
        vertcat(*V): list of states and controls ordered for each node
    """
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



def concat(V, s):
    """Create a single list given a list of lists of variables
    TODO: check size of V elements wrt s, evetually rise size error

        Args:
            V: list of lists of variables
            s: number of nodes

        Returns:
            X: list of all the variables ordered by node
    """

    X = []
    for k in range(s):
        x = []
        for j in V:
            x.append(j[k])
        X.append(vertcat(*x))
    return X

# create_state_and_control: creates state list X and control list U  from VX list and VU list using concat
def create_state_and_control(VX, VU):
    """Create lists of states and controls given two lists containing lists of variables

            Args:
                VX: list of lists of state variables
                VU: list of lists of control variables

            Returns:
                X: list of all the state variables ordered by node
                U: list of all the control variables ordered by node
        """

    ns = np.size(VX[0])
    X = concat(VX, ns)

    ns = np.size(VU[0])
    U = concat(VU, ns)
    return X, U


def dynamic_model_with_floating_base(q, ndot, nddot):
    """
    Construct the floating-base dynamic model:
                x = [q, ndot]
                xdot = [qdot, nddot]
    using quaternion dynamics: quatdot = quat x [omega, 0]
    NOTE: this implementation consider floating-base position and orientation expressed in GLOBAL (world) coordinates while
    linear and angular velocities expressed in LOCAL (base_link) coordinates.
    TODO: creates dedicated file for quaternion handling
    Args:
        q: joint space coordinates: q = [x y z px py pz pw qj], where p is a quaternion
        ndot: joint space velocities: ndot = [vx vy vz wx wy wz qdotj]
        nddot: joint space acceleration: nddot = [ax ay ax wdotx wdoty wdotz qddotj]

    Returns:
        x: state x = [q, ndot]
        xdot: derivative of the state xdot = [qdot, nddot]
    """

    def skew(q):
        """
        Create skew matrix from vector part of quaternion
        TODO: move out
        Args:
            q: vector part of quaternion [qx, qy, qz]

        Returns:
            S = skew symmetric matrix built using q
        """
        S = SX.zeros(3, 3)
        S[0, 1] = -q[2]; S[0, 2] = q[1]
        S[1, 0] = q[2];  S[1, 2] = -q[0]
        S[2, 0] = -q[1]; S[2, 1] = q[0]
        return S

    def quaterion_product(q, p):
        """
        Computes quaternion product between two quaternions q and p
        TODO: move out
        Args:
            q: quaternion
            p: quaternion

        Returns:
            quaternion product q x p
        """
        q0 = q[3]
        p0 = p[3]

        return [q0*p[0:3] + p0*q[0:3] + mtimes(skew(q[0:3]), p[0:3]), q0*p0 - mtimes(q[0:3].T, p[0:3])]

    def toRot(q):
        """
        Compute rotation matrix associated to given quaternion q
        TODO: move out
        Args:
            q: quaternion

        Returns:
            R: rotation matrix

        """
        R = SX.zeros(3, 3)
        qi = q[0]; qj = q[1]; qk = q[2]; qr = q[3]
        R[0, 0] = 1. - 2. * (qj * qj + qk * qk);
        R[0, 1] = 2. * (qi * qj - qk * qr);
        R[0, 2] = 2. * (qi * qk + qj * qr)
        R[1, 0] = 2. * (qi * qj + qk * qr);
        R[1, 1] = 1. - 2. * (qi * qi + qk * qk);
        R[1, 2] = 2. * (qj * qk - qi * qr)
        R[2, 0] = 2. * (qi * qk - qj * qr);
        R[2, 1] = 2. * (qj * qk + qi * qr);
        R[2, 2] = 1. - 2. * (qi * qi + qj * qj)

        return R

    qw = SX.zeros(4,1)
    qw[0:3] = 0.5*ndot[3:6]
    quaterniondot = quaterion_product(q[3:7], qw)

    R = toRot(q[3:7])

    x = vertcat(q, ndot)
    xdot = vertcat(mtimes(R, ndot[0:3]), vertcat(*quaterniondot), ndot[6:ndot.shape[0]], nddot)

    return x, xdot

def create_variable(name, size, number_of_nodes, type, casadi_type = 'SX'):
    """
    Function to create a list of variables
    Args:
        name: of the variable
        size: number of elements of the variable
        number_of_nodes: number of nodes of the problem
        type: type of the variable. This parameter will define the lenght, in terms of nodes, of the variable, in particular:
                if STATE: the variable lenght will be equal the number of nodes of the problem
                if CONTROL: the variable  lenght will be number_of_nodes-1
                if FINAL_STATE: the variable lenght will be 1
        casadi_type: SX or MX

    Returns:
        SX_var: a variable of type SX with the given input size
        opc_var: a list of casady_type of varibales with the given input size and length

    """
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
            NotImplementedError() if not implemented in derived class
        """
        raise NotImplementedError()

class multiple_shooting_LF(constraint_class):
    """
        Class implementing multiple shooting constraint based on two integrators:
            integral(x0, p) = x1
    """
    def __init__(self, dict, F_start, F_integrator):
        """
        Constructor
        Args:
            dict: dictionary containing
                x0: starting point for the integrator
                p: control action
                time: if present contains the time variable
            F_start: integrator called at node k = 0 to start F_integrator
            F_integrator: integrator which will be used in the constraint (started using F_start)
        """
        self.dict = dict
        self.F_start = F_start
        self.F_integrator = F_integrator

        self.x_prev = []

        self.keys = []
        for key in self.dict:
            self.keys.append(key)

    def virtual_method(self, k):
        if k == 0:
            if 'time' in self.dict: # time is optimized
                if np.size(self.dict['time']) == 1: # only final time is optimized
                    integrator_out = self.F_start(x0=self.dict['x0'][k], p=-self.dict['p'][k],
                                                  time=self.dict['time'][0] / np.size(self.dict['p']))
                else: # intermediate times are control variables
                    integrator_out = self.F_start(x0=self.dict['x0'][k], p=-self.dict['p'][k],
                                                  time=self.dict['time'][k])
            else:
                # time is not optimized
                integrator_out = self.F_start(x0=self.dict['x0'][k], p=-self.dict['p'][k],)


            self.x_prev.append(integrator_out['xf'])


        if 'time' in self.dict: # time is optimized
            if np.size(self.dict['time']) == 1: # only final time is optimized
                integrator_out = self.F_integrator(x0=self.dict['x0'][k], x0_prev=self.x_prev[k], p=self.dict['p'][k],
                                                   time=self.dict['time'][0] / np.size(self.dict['p']))
            else: # intermediate times are control variables
                integrator_out = self.F_integrator(x0=self.dict['x0'][k], x0_prev=self.x_prev[k], p=self.dict['p'][k],
                                                   time=self.dict['time'][k])
        else:
            # time is not optimized
            integrator_out = self.F_integrator(x0=self.dict['x0'][k], x0_prev=self.x_prev[k], p=self.dict['p'][k])

        self.x_prev.append(integrator_out['xf_prev'])

        self.gk = [integrator_out['xf'] - self.dict['x0'][k + 1]]
        self.g_mink = [0] * self.dict['x0'][k + 1].size1()
        self.g_maxk = [0] * self.dict['x0'][k + 1].size1()

class multiple_shooting(constraint_class):
    """
    Class implementing multiple shooting constraint:
        integral(x0, p) = x1
    """
    def __init__(self, dict, F_integrator):
        """
        Constructor
        Args:
            dict: dictionary containing
                x0: starting point for the integrator
                p: control action
                time: if present contains the time variable
            F_integrator: integrator which will be used in the constraint
        """
        self.dict = dict
        self.F_integrator = F_integrator

        self.keys = []
        for key in self.dict:
            self.keys.append(key)

    def virtual_method(self, k):
        """
        Computes the multiple shooting constraint at kth node
        Args:
            k: node

        Returns:
            self.gk = constraint at k
            self.g_mink = lower bounds (0)
            self.g_maxk = upper bounds (0)
        """
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


class unit_norm_quaternion(constraint_class):
    """
    Constraint class which impose that quaternion norm is 1
            """

    def __init__(self, quaternion):
        """
        Constructor
        Args:
            quaternion: quaternion to impose unit norm
        """
        self.quaternion = quaternion

    def virtual_method(self, k):
        """Impose quaternion norm to be one for the kth node
        Args:
            k: node
        """
        self.gk = [norm_2(self.quaternion[k])]
        self.g_mink = [1.]
        self.g_maxk = [1.]

class ordered_cost_function_handler(object):
    def __init__(self):
        self.__cost_function_nodes = {}  # dictionary of cost_function : [node_start, node_goal]

        self.__ordered_cost_function = {}  # dictionary of node : [cost_function_1_node, cost_function_2_node, ...]

    def set_cost_function(self, cost_function, from_node, to_node):
        self.__cost_function_nodes[cost_function] = [from_node, to_node]

    def get_cost_function(self):
        for cost_function in self.__cost_function_nodes:
            nodes = self.__cost_function_nodes[cost_function]
            start_node = nodes[0]
            end_node = nodes[1]

            for k in range(start_node, end_node):
                l = cost_function(k)
                if k not in self.__ordered_cost_function:
                    self.__ordered_cost_function[k] = vertcat(l)
                else:
                    self.__ordered_cost_function[k] = vertcat(self.__ordered_cost_function[k], l)

        J = []
        for k in self.__ordered_cost_function:
            J.append(self.__ordered_cost_function[k])
        return J

class ordered_constraint_handler(object):
    """
    Handler class for constraints. It returns constraints ordered by NODE
    """
    def __init__(self):
        self.__constraint_nodes = {} #dictionary of constraint : [node_start, node_goal]

        self.__ordered_constraint = {} #dictionary of node : [constraint_1_node, constraint_2_node, ...]
        self.__ordered_constraint_min = {}  # dictionary of node : [constraint_1_min_node, constraint_2_min_node, ...]
        self.__ordered_constraint_max = {}  # dictionary of node : [constraint_1_max_node, constraint_2_max_node, ...]


    def set_constraint(self, constraint, from_node, to_node):
        """
        Set a constraint to the handler with start and end node
        Args:
            constraint: class
            from_node: starting node
            to_node: ending node

        Returns:

        """
        self.__constraint_nodes[constraint] = [from_node, to_node]

    def get_constraints(self):
        """
        A list of ordered constraints by NODE
        Returns:
            gg: list of constraints by NODE
            gg_min: list of lower bounds by NODE
            gg_max: list of upper bounds by NODE
        """
        for constraint in self.__constraint_nodes:
            nodes = self.__constraint_nodes[constraint]
            start_node = nodes[0]
            end_node = nodes[1]

            for k in range(start_node, end_node):
                g, gmin, gmax = constraint(k)
                if k not in self.__ordered_constraint:
                    self.__ordered_constraint[k] = vertcat(*g)
                    self.__ordered_constraint_max[k] = gmax
                    self.__ordered_constraint_min[k] = gmin
                else:
                    self.__ordered_constraint[k] = vertcat(self.__ordered_constraint[k], *g)
                    self.__ordered_constraint_max[k] += gmax
                    self.__ordered_constraint_min[k] += gmin

        gg = []
        gg_min = []
        gg_max = []
        for k in self.__ordered_constraint:
            gg.append(self.__ordered_constraint[k])
            gg_min.append(self.__ordered_constraint_min[k])
            gg_max.append(self.__ordered_constraint_max[k])
        return gg, gg_min, gg_max


class constraint_handler(object):
    """Class to handle constraints for the optimal control problem
        Attributes:
            g: list of constraints
            g_min: list of lower bounds
            g_max: list of upper bounds
    """
    def __init__(self):
        self.g = []
        self.g_min = []
        self.g_max = []

    def set_constraint(self, g, g_min, g_max):
        """Add constraints and bounds to constraints and bounds lists
        Args:
            g: list of new constraints
            g_min: list of new lower bounds
            g_max: list of new upper bounds
    """
        self.g += g
        self.g_min += g_min
        self.g_max += g_max

    def get_constraints(self):
        """
        Retrieve all the specified constraints and bounds
                Returns:
                    g: vertical concatenation of all constraints g
                    g_min: list of lower bounds
                    g_max: list of upper bounds
            """

        return vertcat(*self.g), self.g_min, self.g_max

def retrieve_solution(input, output_dict, solution):
    """
    Function which evaluates separately each component of solution.
            Args:
                input: symbolic vector of variables of the optimization problem
                output_dict: how the output dictionary definition
                solution: solution computed by the solver
            Returns:
                o: dictionary defined by output_dict containing solution
    """
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





