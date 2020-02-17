from casadi import *
from horizon import *

def integrator_time(dae_time):

    x = dae_time['x']
    qddot = dae_time['p']
    xdot = dae_time['ode']
    L = dae_time['quad']

    f_RK = Function('f_RK', [x, qddot], [xdot, L])

    nx = x.size1()
    nv = qddot.size1()

    X0_RK = MX.sym('X0_RK', nx)
    U_RK = MX.sym('U_RK', nv)
    DT_RK = MX.sym('DT_RK', 1)
    X_RK = X0_RK
    Q_RK = 0

    k1, k1_q = f_RK(X_RK, U_RK)
    k2, k2_q = f_RK(X_RK + 0.5 * DT_RK * k1, U_RK)
    k3, k3_q = f_RK(X_RK + DT_RK / 2 * k2, U_RK)
    k4, k4_q = f_RK(X_RK + DT_RK * k3, U_RK)
    X_RK = X_RK + DT_RK / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    Q_RK = Q_RK + DT_RK / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

    F_integrator_time = Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])

    return F_integrator_time

