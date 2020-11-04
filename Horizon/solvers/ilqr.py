import casadi as cs
import numpy as np
from horizon.utils import integrator
from typing import List

class IterativeLQR:
    """
    The IterativeLQR class solves a nonlinear, unconstrained iLQR problem for a given
     - system dynamics (continuous time)
     - intermediate cost l(x,u)
     - final cost lf(x)
    """

    class LinearDynamics:

        def __init__(self, nx: int, nu: int):

            self.A = np.zeros((nx, nx))
            self.B = np.zeros((nx, nu))
            self.Fxx = np.zeros((nx*nx, nx))
            self.Fuu = np.zeros((nx*nu, nu))
            self.Fux = np.zeros((nx*nu, nx))

        def __repr__(self):
            return self.__dict__.__repr__()

    class LinearConstraint:

        def __init__(self, nx: int, nu: int, nc: int):

            self.C = np.zeros((nc, nx))
            self.D = np.zeros((nc, nu))
            self.g = np.zeros(nc)

        def __repr__(self):
            return self.__dict__.__repr__()

    class QuadraticCost:

        def __init__(self, nx: int, nu: int):
            self.qx = np.zeros(nx)
            self.Qxx = np.zeros((nx, nx))
            self.qu = np.zeros(nu)
            self.Quu = np.zeros((nu, nu))
            self.Qxu = np.zeros((nx, nu))

        def __repr__(self):
            return self.__dict__.__repr__()

    def __init__(self,
                 xdot: cs.Function,
                 dt: float,
                 N: int,
                 diff_intermediate_cost: cs.Function,
                 final_cost: cs.Function,
                 final_constraint=None,
                 sym_t=cs.MX):
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
        self._sym_t = sym_t
        self._state_trj = [np.zeros(self._nx) for _ in range(self._N + 1)]
        self._ctrl_trj  = [np.zeros(self._nu) for _ in range(self._N)]
        self._lin_dynamics = [self.LinearDynamics(self._nx, self._nu) for _ in range(self._N)]
        self._inter_quad_cost = [self.QuadraticCost(self._nx, self._nu) for _ in range(self._N)]
        self._final_quad_cost = self.QuadraticCost(self._nx, 0)
        self._cost_to_go = [self.QuadraticCost(self._nx, self._nu) for _ in range(self._N)]
        self._value_function = [self.QuadraticCost(self._nx, self._nu) for _ in range(self._N)]
        self._diff_inter_cost = diff_intermediate_cost
        self._final_cost = final_cost
        self._final_constraint = final_constraint
        self._constraint_to_go = None

        if final_constraint is not None:

            if final_constraint.name_out(0) != 'gf':
                raise KeyError('Final constraint output must be named "gf"')

            self._nc = final_constraint.size1_out('gf')
            self._constraint_to_go = self.LinearConstraint(self._nx, self._nu, self._nc)
            self._final_constraint_jac = self._final_constraint.jac()

        self._dynamics_ct = xdot
        self._dt = dt
        self._fb_gain = [np.zeros((self._nu, self._nx)) for _ in range(self._N)]
        self._ff_u = [np.zeros(self._nu) for _ in range(self._N)]
        self._defect = [np.zeros(self._nx) for _ in range(self._N)]

        self._defect_norm = []
        self._du_norm = []
        self._dx_norm = []
        self._dcost = []

        self._use_second_order_dynamics = False
        self._use_single_shooting_state_update = False

        self._discretize()

    def _make_jit_function(f: cs.Function):
        """
        Compiles casadi function into a shared object and return it
        :return:
        """

        import filecmp
        import os

        gen_code_path = 'ilqr_generated_{}.c'.format(f.name())
        f.generate(gen_code_path)

        gen_lib_path = 'ilqr_generated_{}.so'.format(f.name())
        gcc_cmd = 'gcc {} -shared -fPIC -O3 -ffast-math -o {}'.format(gen_code_path, gen_lib_path)

        if os.system(gcc_cmd) != 0:
            raise SystemError('Unable to compile function "{}"'.format(f.name()))

        jit_f = cs.external(f.name(), './' + gen_lib_path)

        os.remove(gen_code_path)
        os.remove(gen_lib_path)

        return jit_f


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
        # self._F = cs.Function('F',
        #                       {'x0': x, 'p': u,
        #                        'xf': x + self._dt * self._dynamics_ct(x, u),
        #                        'qf': self._dt * self._diff_inter_cost(x, u)
        #                        },
        #                       ['x0', 'p'],
        #                       ['xf', 'qf'])

        self._F = integrator.RK4(dae, {'tf': self._dt}, 'MX')
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

            for j in range(self._nx):
                nx = self._nx
                nu = self._nu
                self._lin_dynamics[i].Fxx[j*nx:(j+1)*nx, :] = hode_value['DDxfDx0Dx0'].toarray()[j::nx, :]
                self._lin_dynamics[i].Fuu[j*nu:(j+1)*nu, :] = hode_value['DDxfDpDp'].toarray()[j::nx, :]
                self._lin_dynamics[i].Fux[j*nu:(j+1)*nu, :] = hode_value['DDxfDpDx0'].toarray()[j::nx, :]

        if self._final_constraint is not None:

            jgf_value = self._final_constraint_jac(x=self._state_trj[-1])
            self._constraint_to_go = self.LinearConstraint(self._nx, self._nu, self._nc)
            self._constraint_to_go.C = jgf_value['DgfDx'].toarray()
            self._constraint_to_go.D = np.zeros((self._nc, self._nu))
            self._constraint_to_go.g = self._final_constraint(x=self._state_trj[-1])['gf'].toarray().flatten()

    def _backward_pass(self):
        """
        To be implemented
        :return:
        """

        # value function at next time step (prev iteration)
        S = self._final_quad_cost.Qxx
        s = self._final_quad_cost.qx

        for i in reversed(range(self._N)):

            # variable labeling for better convenience
            nx = self._nx
            nu = self._nu
            x_integrated = self._F(x0=self._state_trj[i], p=self._ctrl_trj[i])['xf'].toarray().flatten()
            xnext = self._state_trj[i+1]
            d = x_integrated - xnext
            r = self._inter_quad_cost[i].qu
            q = self._inter_quad_cost[i].qx
            P = self._inter_quad_cost[i].Qxu.T
            R = self._inter_quad_cost[i].Quu
            Q = self._inter_quad_cost[i].Qxx
            A = self._lin_dynamics[i].A
            B = self._lin_dynamics[i].B
            Fxx = self._lin_dynamics[i].Fxx.reshape((nx, nx, nx))
            Fuu = self._lin_dynamics[i].Fuu.reshape((nx, nu, nu))
            Fux = self._lin_dynamics[i].Fux.reshape((nx, nu, nx))

            # constraint handling
            constrained = self._constraint_to_go is not None and self._constraint_to_go.g.size > 0
            l_ff = np.zeros(self._nu)
            L_fb = np.zeros((self._nu, self._nx))
            Vns = np.eye(self._nu)

            if constrained:

                # back-propagate constraint to go from next time step
                C = self._constraint_to_go.C@A
                D = self._constraint_to_go.C@B
                g = self._constraint_to_go.g - self._constraint_to_go.C@d

                # svd of constraint input matrix
                U, sv, V = np.linalg.svd(D)
                V = V.T

                # rotated constraint
                rot_g = U.T @ g
                rot_C = U.T @ C

                # non-zero singular values
                large_sv = sv > 1e-4

                nc = g.size  # number of currently active constraints
                nsv = len(sv)  # number of singular values
                rank = np.count_nonzero(large_sv)  # constraint input matrix rank

                # singular value inversion
                inv_sv = sv.copy()
                inv_sv[large_sv] = np.reciprocal(sv[large_sv])

                # compute constraint component of control input uc = Lc*x + lc
                l_ff = -V[:, 0:nsv] @ (inv_sv*rot_g[0:nsv])
                l_ff.flatten()
                L_fb = -V[:, 0:nsv] @ np.diag(inv_sv) @ rot_C[0:nsv, :]

                # update constraint to go
                left_constraint_dim = nc - rank

                if left_constraint_dim == 0:
                    self._constraint_to_go = None
                else:
                    self._constraint_to_go.C = rot_C[rank:, :]
                    self._constraint_to_go.D = np.zeros((left_constraint_dim, self._nu))
                    self._constraint_to_go.g = rot_g[rank:]

                nullspace_dim = self._nu - rank

                if nullspace_dim == 0:
                    Vns = np.zeros((self._nu, 0))
                else:
                    Vns = V[:, -nullspace_dim:]

                # the constraint induces a modified dynamics via u = Lx + l + Vns*z (z = new control input)
                d = d + B@l_ff
                A = A + B@L_fb
                B = B@Vns

                if self._use_second_order_dynamics:

                    tr_idx = (0, 2, 1)

                    d += 0.5*l_ff@Fuu@l_ff
                    A += l_ff@Fuu@L_fb + l_ff@Fux
                    B += l_ff@Fuu@Vns

                    Fxx = Fxx + L_fb.T @ (Fuu @ L_fb + Fux) + Fux.transpose(tr_idx)@L_fb
                    Fux = Vns.T @ (Fux + Fuu @ L_fb)
                    Fuu = Vns.T @ Fuu @ Vns

                q = q + L_fb.T@(r + R@l_ff) + P.T@l_ff
                Q = Q + L_fb.T@R@L_fb + L_fb.T@P + P.T@L_fb
                P = Vns.T @ (P + R@L_fb)

                r = Vns.T @ (r + R @ l_ff)
                R = Vns.T @ R @ Vns

            # intermediate quantities
            hx = q + A.T@(s + S@d)
            hu = r + B.T@(s + S@d)
            Huu = R + B.T@S@B
            Hux = P + B.T@S@A
            Hxx = Q + A.T@S@A

            if self._use_second_order_dynamics:

                Huu += (Fuu.T @ (s + S@d)).T
                Hux += (Fux.T @ (s + S@d)).T
                Hxx += (Fxx.T @ (s + S@d)).T


            # nullspace gain and feedforward computation
            lam_min = 0

            if Huu.size > 0:
                lam_Huu = np.linalg.eigvalsh(R)
                cond_max = 1000
                lam_min = lam_Huu.min()
                lam_max = lam_Huu.max()

            if lam_min < 0:
                print(lam_Huu)
                eps = (lam_max + abs(lam_min))/cond_max + abs(lam_min)
                Huu += np.eye(nu) * eps
                lam_min = 0
                lam_Huu += lam_min

            l_Lz = -np.linalg.solve(Huu, np.hstack((hu.reshape((hu.size, 1)), Hux)))
            lz = l_Lz[:, 0]
            Lz = l_Lz[:, 1:]

            # overall gain and ffwd including constraint
            l_ff = l_ff + Vns @ lz
            L_fb = L_fb + Vns @ Lz

            # value function update
            # Atil = A + B@Lz
            # dtil = d + B@lz
            # s = q + Atil.T@(S@dtil + s) + Lz.T@(r + R@lz) + P.T@lz
            # S = Q + Atil.T@S@Atil + Lz.T@R@Lz + Lz.T@P + P.T@Lz
            s = hx - Lz.T@Huu@lz
            S = Hxx - Lz.T@Huu@Lz

            # save gain and ffwd
            self._fb_gain[i] = L_fb.copy()
            self._ff_u[i] = l_ff.copy()

            # save defect (for original dynamics)
            d = x_integrated - xnext
            self._defect[i] = d.copy()

    class PropagateResult:
        def __init__(self):
            self.state_trj = []
            self.ctrl_trj = []
            self.dx_norm = 0.0
            self.du_norm = 0.0
            self.cost = 0.0

    def _forward_pass(self):
        """
        To be implemented
        :return:
        """
        x_old = self._state_trj.copy()

        defect_norm = 0
        du_norm = 0
        dx_norm = 0

        for i in range(self._N):

            xnext = self._state_trj[i+1]
            xi_upd = self._state_trj[i]
            ui = self._ctrl_trj[i]
            d = self._defect[i]
            A = self._lin_dynamics[i].A
            B = self._lin_dynamics[i].B
            L = self._fb_gain[i]
            l = self._ff_u[i]
            dx = np.atleast_1d(xi_upd - x_old[i])

            ui_upd = ui + l + L@dx

            if self._use_single_shooting_state_update:
                xnext_upd = self._F(x0=xi_upd, p=ui_upd)['xf'].toarray().flatten()
            else:
                xnext_upd = xnext + (A + B@L)@dx + B@l + d


            self._state_trj[i+1] = xnext_upd.copy()
            self._ctrl_trj[i] = ui_upd.copy()

            defect_norm += np.linalg.norm(d, ord=1)
            du_norm += np.linalg.norm(l, ord=1)
            dx_norm += np.linalg.norm(dx, ord=1)


        self._defect_norm.append(defect_norm)
        self._du_norm.append(du_norm)
        self._dx_norm.append(dx_norm)
        self._dcost.append(self._eval_cost(self._state_trj, self._ctrl_trj))

    def _propagate(self, xtrj: List[np.array], utrj: List[np.array], alpha=1):

        N = len(utrj)
        ret = self.PropagateResult()

        ret.state_trj = xtrj.copy()
        ret.ctrl_trj = utrj.copy()

        for i in range(N):

            xnext = xtrj[i+1]
            xi = xtrj[i]
            xi_upd = ret.state_trj[i]
            ui = utrj[i]
            d = self._defect[i]
            A = self._lin_dynamics[i].A
            B = self._lin_dynamics[i].B
            L = self._fb_gain[i]
            l = alpha * self._ff_u[i]
            dx = np.atleast_1d(xi_upd - xi)

            ui_upd = ui + l + L@dx

            if self._use_single_shooting_state_update:
                xnext_upd = self._F(x0=xi_upd, p=ui_upd)['xf'].toarray().flatten()
            else:
                xnext_upd = xnext + (A + B@L)@dx + B@l + d

            ret.state_trj[i+1] = xnext_upd.copy()
            ret.ctrl_trj[i] = ui_upd.copy()
            ret.dx_norm += np.linalg.norm(dx, ord=1)
            ret.du_norm += np.linalg.norm(ui_upd - ui, ord=1)

        ret.cost = self._eval_cost(ret.state_trj, ret.ctrl_trj)

        return ret


    def _eval_cost(self, x_trj, u_trj):

        cost = 0.0

        for i in range(len(u_trj)):

            cost += self._F(x0=x_trj[i], p=u_trj[i])['qf'].__float__()

        cost += self._final_cost(x=x_trj[-1])['l'].__float__()

        return cost


    def solve(self, niter: int):

        if len(self._dcost) == 0:
            self._dcost.append(self._eval_cost(self._state_trj, self._ctrl_trj))

        for _ in range(niter):

            self._linearize_quadratize()
            self._backward_pass()
            self._forward_pass()

    def setInitialState(self, x0: np.array):

        self._state_trj[0] = np.array(x0)

    def randomizeInitialGuess(self):

        self._state_trj[1:] = [np.random.randn(self._nx) for _ in range(self._N)]
        self._ctrl_trj  = [np.random.randn(self._nu) for _ in range(self._N)]

        if self._use_single_shooting_state_update:
            for i in range(self._N):
                self._state_trj[i+1] = self._F(x0=self._state_trj[i], p=self._ctrl_trj[i])['xf'].toarray().flatten()




