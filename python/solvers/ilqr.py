import casadi as cs
import numpy as np

class IterativeLQR:
    """
    The IterativeLQR class solves a nonlinear, unconstrained iLQR problem for a given
     - system dynamics (continuous time)
     - intermediate cost l(x,u)
     - final cost lf(x)
    """

    class LinearDynamics:

        def __init__(self, nx: int, nu: int):

            self.A = np.array((nx, nx))
            self.B = np.array((nx, nu))

        def __repr__(self):
            return self.__dict__.__repr__()

    class QuadraticCost:

        def __init__(self, nx: int, nu: int):
            self.qx = np.array(nx)
            self.Qxx = np.array((nx, nx))
            self.qu = np.array(nu)
            self.Quu = np.array((nu, nu))
            self.Qxu = np.array((nx, nu))


    def __init__(self,
                 xdot: cs.Function,
                 dt: float,
                 N: int,
                 diff_intermediate_cost: cs.Function,
                 final_cost: cs.Function):
        """
        Constructor
        :param xdot: casadi.Function with inputs named x, u; output is named xdot.
        This models the system continuous time dynamics
        :param diff_intermediate_cost: casadi.Function with inputs named x, u; output is named l
        :param final_cost: casadi.Function with input named x; output is named l
        """


        if diff_intermediate_cost.name_in(0) != 'x':
            raise KeyError('First input of l must be named "x"')

        if final_cost.name_in(0) != 'x':
            raise KeyError('First input of lf must be named "x"')

        if diff_intermediate_cost.name_in(1) != 'u':
            raise KeyError('Second input of l must be named "u"')

        self._nx = final_cost.size1_in('x')
        self._nu = diff_intermediate_cost.size1_in('u')
        self._jacobian_lf = final_cost.jac()
        self._hessian_lf = self._jacobian_lf.jac()
        self._N = N
        self._sym_t = cs.MX
        self._state_trj = [np.zeros(self._nx) for _ in range(self._N + 1)]
        self._ctrl_trj  = [np.zeros(self._nu) for _ in range(self._N)]
        self._lin_dynamics = [self.LinearDynamics(self._nx, self._nu) for _ in range(self._N)]
        self._inter_quad_cost = [self.QuadraticCost(self._nx, self._nu) for _ in range(self._N)]
        self._final_quad_cost = self.QuadraticCost(self._nx, 0)
        self._diff_inter_cost = diff_intermediate_cost
        self._final_cost = final_cost
        self._dynamics_ct = xdot
        self._dt = dt
        self._fb_gain = [np.zeros((self._nu, self._nx)) for _ in range(self._N)]
        self._ff_u = [np.zeros(self._nu) for _ in range(self._N)]
        self._defect = [np.zeros(self._nx) for _ in range(self._N)]

        self._discretize()
        self._linearize_quadratize()


    def _discretize(self):
        """
        Compute discretized dynamics in the form of _F (nonlinear state transition function) and
        _jacobian_F (its jacobian)
        :return: None
        """

        x = self._sym_t.sym('x', self._nx)
        u = self._sym_t.sym('u', self._nu)

        dae = {'x': x,
               'p': u,
               'ode': self._dynamics_ct(x, u),
               'quad': self._diff_inter_cost(x, u)}

        # self._F = cs.integrator('F', 'rk', dae, {'t0': 0, 'tf': self._dt})
        self._F = cs.Function('F',
                              {'x0': x, 'p': u,
                               'xf': x + self._dt * self._dynamics_ct(x, u),
                               'qf': self._dt * self._diff_inter_cost(x, u)
                               },
                              ['x0', 'p'],
                              ['xf', 'qf'])


        self._jacobian_F = self._F.jac()
        self._hessian_F = self._jacobian_F.jac()




    def _linearize_quadratize(self):
        """
        Compute quadratic approximations to cost functions about the current state and control trajectories
        :return: None
        """

        jl_value = self._jacobian_lf(x=self._state_trj[-1])

        hl_value = self._hessian_lf(x=self._state_trj[-1])

        self._final_quad_cost.qx = jl_value['DlDx'].toarray().flatten()
        self._final_quad_cost.Qxx = hl_value['DDlDxDx'].toarray()


        for i in range(self._N):

            jode_value = self._jacobian_F(x0=self._state_trj[i],
                                          p=self._ctrl_trj[i])

            hode_value = self._hessian_F(x0=self._state_trj[i],
                                         p=self._ctrl_trj[i])


            self._inter_quad_cost[i].qu = jode_value['DqfDp'].toarray().flatten()
            self._inter_quad_cost[i].qx = jode_value['DqfDx0'].toarray().flatten()
            self._inter_quad_cost[i].Quu = hode_value['DDqfDpDp'].toarray()
            self._inter_quad_cost[i].Qxx = hode_value['DDqfDx0Dx0'].toarray()
            self._inter_quad_cost[i].Qxu = hode_value['DDqfDx0Dp'].toarray()

            self._lin_dynamics[i].A = jode_value['DxfDx0'].toarray()
            self._lin_dynamics[i].B = jode_value['DxfDp'].toarray()



    def _backward_pass(self):
        """
        To be implemented
        :return:
        """

        S = self._final_quad_cost.Qxx
        s = self._final_quad_cost.qx

        for i in reversed(range(self._N)):

            # variable labeling for better convenience
            x_integrated = self._F(x0=self._state_trj[i], p=self._ctrl_trj[i])['xf'].toarray().flatten()
            xnext = self._state_trj[i+1]
            r = self._inter_quad_cost[i].qu
            q = self._inter_quad_cost[i].qx
            P = self._inter_quad_cost[i].Qxu.T
            R = self._inter_quad_cost[i].Quu
            Q = self._inter_quad_cost[i].Qxx
            A = self._lin_dynamics[i].A
            B = self._lin_dynamics[i].B

            # back propagation
            d = x_integrated - xnext
            h = r + B.T @ (s + S@d)
            G = P + B.T @ S @ A
            H = R + B.T @ S @ B

            # gain and feedforward computation
            l_L = -np.linalg.solve(H, np.hstack((h.reshape((h.size, 1)), G)))
            l = l_L[:, 0]
            L = l_L[:, 1:]

            # value function update
            s = q + A.T @ (s + S@d) + G.T @ l + L.T @ (h + H@l)
            S = Q + A.T @ S @ A - L.T @ H @ L

            # save gain and ffwd
            self._fb_gain[i] = L.copy()
            self._ff_u[i] = l.copy()
            self._defect[i] = d.copy()



    def _forward_pass(self):
        """
        To be implemented
        :return:
        """
        x_old = self._state_trj.copy()

        for i in range(self._N):

            xnext = self._state_trj[i+1]
            xi_upd = self._state_trj[i]
            ui = self._ctrl_trj[i]
            d = self._defect[i]
            A = self._lin_dynamics[i].A
            B = self._lin_dynamics[i].B
            L = self._fb_gain[i]
            l = self._ff_u[i]
            dx = xi_upd - x_old[i]

            xnext_upd = xnext + (A + B@L)@dx + B@l + d
            ui_upd = ui  + l + L@dx

            self._state_trj[i+1] = xnext_upd.copy()
            self._ctrl_trj[i] = ui_upd.copy()


    def solve(self, niter: int):

        for _ in range(niter):

            self._linearize_quadratize()
            self._backward_pass()
            self._forward_pass()

    def setInitialState(self, x0: np.array):

        self._state_trj[0] = np.array(x0)


