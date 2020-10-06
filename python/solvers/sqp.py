from casadi import *
from numpy import *

def sqpsol(name, problem_dict, options_dict):
    """
    sqpsol creates a sqp solver
    Args:
        name: name of the solver (not used at the moment)
        problem_dict: {'f': cost_function, 'g': constraints, 'x': variables}
        options_dict: {'max_iter': iterations, 'qpsolver': internal_qpsolver

    Returns: a sqp object

    """
    return sqp(name, problem_dict, options_dict)

class sqp(object):
    """
    Implements a sqp solver
    """
    def __init__(self, name, problem_dict, options_dict):
        """

        Args:
            name: name of the solver (not used at the moment)
            problem_dict: {'f': cost_function, 'g': constraints, 'x': variables}
            options_dict: {'max_iter': iterations, 'qpsolver': internal_qpsolver
        """
        self.__name = name
        self.__problem_dict = problem_dict
        self.__options_dict = options_dict

        self.__f = self.__problem_dict['f']
        self.__g = self.__problem_dict['g']
        self.__x = self.__problem_dict['x']

        self.__max_iter = 1000
        if 'max_iter' in options_dict:
            self.__max_iter = options_dict['max_iter']

        self.__qpsolver = "qpoases"
        if 'qpsolver' in options_dict:
            self.__qpsolver = options_dict['qpsolver']

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

    def qpsolve(self, H, g, lbx, ubx, A, lba, uba):
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

        # QP structure
        qp = {}
        qp['h'] = H.sparsity()
        qp['a'] = A.sparsity()

        # Create CasADi solver instance
        solver = conic('S', self.__qpsolver, qp)

        r = solver(h=H, g=g, a=A, lbx=lbx, ubx=ubx, lba=lba, uba=uba)

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

        self.__v0 = x0
        self.__vmin = lbx
        self.__vmax = ubx
        self.__gmin = lbg
        self.__gmax = ubg

        self.__v_opt = self.__v0
        for k in range(self.__max_iter):
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
            dv = self.qpsolve(H_k, Grad_obj_k, dv_min, dv_max, J_g_k, -g_k, -g_k)

            # Take the full step
            self.__v_opt += dv.toarray().flatten()
            self.__obj.append(float(dot(r_k.T, r_k) / 2))
            self.__constr.append(float(norm_2(g_k)))

        solution_dict = {'x': self.__v_opt, 'f': self.__obj, 'g': self.__constr}
        return solution_dict