from casadi import *

def sqpsol(name, qp_solver, problem_dict, options_dict):
    """
    sqpsol creates a sqp solver
    Args:
        name: name of the solver (not used at the moment)
        qp_solver: internal qp solver name
        problem_dict: {'f': cost_function, 'g': constraints, 'x': variables}
        options_dict: {'max_iter': iterations}

    Returns: a sqp object

    """
    return sqp(name, qp_solver, problem_dict, options_dict)

def qpoasesMPCOptions():
    opts = {'qpoases.sparse': True,
            'qpoases.linsol_plugin': 'ma57',
            'qpoases.enableRamping': False,
            'qpoases.enableFarBounds': False,
            'qpoases.enableFlippingBounds': False,
            'qpoases.enableFullLITests': False,
            'qpoases.enableNZCTests': False,
            'qpoases.enableDriftCorrection': 0,
            'qpoases.enableCholeskyRefactorisation': 0,
            'qpoases.enableEqualities': True,
            'qpoases.initialStatusBounds': 'inactive',
            'qpoases.numRefinementSteps': 0,
            'qpoases.terminationTolerance': 1e9 * np.finfo(float).eps,
            'qpoases.enableInertiaCorrection': False,
            'qpoases.printLevel': 'none'}
    return opts

class sqp(object):
    """
    Implements a sqp solver
    """
    def __init__(self, name, qp_solver, problem_dict, options_dict):
        """

        Args:
            name: name of the solver (not used at the moment)
            problem_dict: {'f': residual of cost function, 'g': constraints, 'x': variables}
            options_dict: {'max_iter': iterations, 'qpsolver': internal_qpsolver
        """
        self.__name = name
        self.__problem_dict = problem_dict
        self.__options_dict = options_dict
        self.__qpsolver = qp_solver

        self.__qpsolver_options = self.qpsolver_option_parser(self.__qpsolver, self.__options_dict)

        self.__f = self.__problem_dict['f']
        self.__g = self.__problem_dict['g']
        self.__x = self.__problem_dict['x']

        self.__max_iter = 1000
        if 'max_iter' in self.__options_dict:
            self.__max_iter = self.__options_dict['max_iter']

        self.__reinitialize_qpsolver = False
        if 'reinitialize_qpsolver' in self.__options_dict:
            self.__reinitialize_qpsolver = self.__options_dict['reinitialize_qpsolver']




        # Form function for calculating the Gauss-Newton objective
        self.__r_fcn = Function('r_fcn', {'v': self.__x, 'r': self.__f}, ['v'], ['r'])

        # Form function for calculating the constraints
        self.__g_fcn = Function('g_fcn', {'v': self.__x, 'g': self.__g}, ['v'], ['g'])

        # Generate functions for the Jacobians
        self.__Jac_r_fcn = self.__r_fcn.jac()
        self.__Jac_g_fcn = self.__g_fcn.jac()

        self.__v0 = []
        self.__vmin = []
        self.__vmax = []
        self.__gmin = []
        self.__gmax = []

        self.__v_opt = []
        self.__obj = []
        self.__constr = []

        self.__solver = []
        self.__qp = {}

    def qpsolve(self, H, g, lbx, ubx, A, lba, uba, init=True):
        """
        Internal qp solver to solve differential problem
        Args:
            H: Hessian cos function
            g: gradient cost function
            lbx: lower bounds
            ubx: upper bounds
            A: Constraints
            lba: lower constraints
            uba: upper constraints

        Returns: solution of differential problem

        """

        if init:
            # QP structure
            self.__qp['h'] = H.sparsity()
            self.__qp['a'] = A.sparsity()


            # Create CasADi solver instance
            self.__solver = conic('S', self.__qpsolver, self.__qp, self.__qpsolver_options)

        r = self.__solver(h=H, g=g, a=A, lbx=lbx, ubx=ubx, lba=lba, uba=uba)

        # Return the solution
        return r['x']

    def __call__(self, x0, lbx, ubx, lbg, ubg):
        """
        Compute solution of non linear problem
        Args:
            x0: initial guess
            lbx: lower bounds
            ubx: upper bounds
            lbg: lower constraints
            ubg: upper constraints

        Returns: solution dict {'x': nlp_solution, 'f': value_cost_function, 'g': value_constraints}

        """
        from numpy import *

        self.__v0 = x0
        self.__vmin = lbx
        self.__vmax = ubx
        self.__gmin = lbg
        self.__gmax = ubg

        self.__v_opt = self.__v0
        for k in range(self.__max_iter):

            init = self.__reinitialize_qpsolver
            if k == 0:
                init = True

            # Form quadratic approximation of objective
            Jac_r_fcn_value = self.__Jac_r_fcn(v=self.__v_opt)  # evaluate in v_opt
            J_r_k = Jac_r_fcn_value['DrDv']
            r_k = self.__r_fcn(v=self.__v_opt)['r']

            # Form quadratic approximation of constraints
            Jac_g_fcn_value = self.__Jac_g_fcn(v=self.__v_opt)  # evaluate in v_opt
            J_g_k = Jac_g_fcn_value['DgDv']
            g_k = self.__g_fcn(v=self.__v_opt)['g']

            # Gauss-Newton Hessian
            H_k = mtimes(J_r_k.T, J_r_k)

            # Gradient of the objective function
            Grad_obj_k = mtimes(J_r_k.T, r_k)

            # Bounds on delta_v
            dv_min = self.__vmin - self.__v_opt
            dv_max = self.__vmax - self.__v_opt

            # Solve the QP
            dv = self.qpsolve(H_k, Grad_obj_k, dv_min, dv_max, J_g_k, -g_k, -g_k, init)

            # Take the full step
            self.__v_opt += 0.5*dv.toarray().flatten()
            self.__obj.append(float(dot(r_k.T, r_k) / 2.))
            self.__constr.append(float(norm_2(g_k)))

        solution_dict = {'x': self.__v_opt, 'f': self.__obj, 'g': self.__constr}
        return solution_dict

    def qpsolver_option_parser(self, qpsolver, options):
        parsed_options = {}
        for key in options:
            list = key.split(".")
            if list[0] == qpsolver:
                parsed_options[list[1]] = options[key]
        return parsed_options

    def plot(self):
        import matplotlib.pyplot as plt
        # Plot the results
        plt.figure(1)

        plt.title("SQP solver output")
        plt.semilogy(self.__obj)
        plt.semilogy(self.__constr)
        plt.xlabel('iteration')
        plt.legend(['Objective value', 'Constraint violation'], loc='center right')
        plt.grid()

        plt.show()