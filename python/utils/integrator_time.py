from casadi import *
from horizon import *

class integrator_time:
    def __init__(self, dae_time):
        self.x = dae_time['x']
        self.qddot = dae_time['p']
        self.xdot = dae_time['ode']
        self.L = dae_time['quad']

        self.f_RK = Function('f_RK', [self.x, self.qddot], [self.xdot, self.L])

        nx = self.x.size1()
        nv = self.qddot.size1()

        self.X0_RK = MX.sym('X0_RK', nx)
        self.U_RK = MX.sym('U_RK', nv)
        self.DT_RK = MX.sym('DT_RK', 1)
        self.X_RK = self.X0_RK
        self.Q_RK = 0

        self.k1, self.k1_q = self.f_RK(self.X_RK, self.U_RK)
        self.k2, self.k2_q = self.f_RK(self.X_RK + 0.5 * self.DT_RK * self.k1, self.U_RK)
        self.k3, self.k3_q = self.f_RK(self.X_RK + self.DT_RK / 2. * self.k2, self.U_RK)
        self.k4, self.k4_q = self.f_RK(self.X_RK + self.DT_RK * self.k3, self.U_RK)
        self.X_RK = self.X_RK + self.DT_RK / 6. * (self.k1 + 2. * self.k2 + 2. * self.k3 + self.k4)
        self.Q_RK = self.Q_RK + self.DT_RK / 6. * (self.k1_q + 2. * self.k2_q + 2. * self.k3_q + self.k4_q)

        self.F_integrator_time = Function('F_RK', [self.X0_RK, self.U_RK, self.DT_RK], [self.X_RK, self.Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])

    def getIntegrator(self):
        return self.F_integrator_time

