from casadi import *
from horizon import *

def RK4(dae, opts, casadi_type):
    x = dae['x']
    qddot = dae['p']
    xdot = dae['ode']
    L = dae['quad']

    f_RK = Function('f_RK', [x, qddot], [xdot, L])

    nx = x.size1()
    nv = qddot.size1()

    if casadi_type is 'MX':
        X0_RK = MX.sym('X0_RK', nx)
        U_RK = MX.sym('U_RK', nv)
    elif casadi_type is 'SX':
        X0_RK = SX.sym('X0_RK', nx)
        U_RK = SX.sym('U_RK', nv)
    else:
        raise Exception('Input casadi_type can be only SX or MX!')

    DT_RK = opts['tf']
    X_RK = X0_RK
    Q_RK = 0

    k1, k1_q = f_RK(X_RK, U_RK)
    k2, k2_q = f_RK(X_RK + 0.5 * DT_RK * k1, U_RK)
    k3, k3_q = f_RK(X_RK + DT_RK / 2. * k2, U_RK)
    k4, k4_q = f_RK(X_RK + DT_RK * k3, U_RK)

    X_RK = X_RK + DT_RK / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    Q_RK = Q_RK + DT_RK / 6. * (k1_q + 2. * k2_q + 2. * k3_q + k4_q)

    return Function('F_RK', [X0_RK, U_RK], [X_RK, Q_RK], ['x0', 'p'], ['xf', 'qf'])

def RK4_time(dae, casadi_type):
    x = dae['x']
    qddot = dae['p']
    xdot = dae['ode']
    L = dae['quad']

    f_RK = Function('f_RK', [x, qddot], [xdot, L])

    nx = x.size1()
    nv = qddot.size1()

    if casadi_type is 'MX':
        X0_RK = MX.sym('X0_RK', nx)
        U_RK = MX.sym('U_RK', nv)
        DT_RK = MX.sym('DT_RK', 1)
    elif casadi_type is 'SX':
        X0_RK = SX.sym('X0_RK', nx)
        U_RK = SX.sym('U_RK', nv)
        DT_RK = SX.sym('DT_RK', 1)
    else:
        raise Exception('Input casadi_type can be only SX or MX!')

    X_RK = X0_RK
    Q_RK = 0

    k1, k1_q = f_RK(X_RK, U_RK)
    k2, k2_q = f_RK(X_RK + 0.5 * DT_RK * k1, U_RK)
    k3, k3_q = f_RK(X_RK + DT_RK / 2. * k2, U_RK)
    k4, k4_q = f_RK(X_RK + DT_RK * k3, U_RK)

    X_RK = X_RK + DT_RK / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    Q_RK = Q_RK + DT_RK / 6. * (k1_q + 2. * k2_q + 2. * k3_q + k4_q)

    return Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])

def RKF45(dae, opts, casadi_type):

    x = dae['x']
    qddot = dae['p']
    xdot = dae['ode']
    L = dae['quad']

    f_RK = Function('f_RK', [x, qddot], [xdot, L])

    nx = x.size1()
    nv = qddot.size1()

    if casadi_type is 'MX':
        X0_RK = MX.sym('X0_RK', nx)
        U_RK = MX.sym('U_RK', nv)
    elif casadi_type is 'SX':
        X0_RK = SX.sym('X0_RK', nx)
        U_RK = SX.sym('U_RK', nv)
    else:
        raise Exception('Input casadi_type can be only SX or MX!')

    DT_RK = opts['tf']
    X_RK = X0_RK
    Q_RK = 0

    k1, k1_q = f_RK(X_RK, U_RK)

    k2, k2_q = f_RK(X_RK + k1 * (DT_RK / 5),
                    U_RK)

    k3, k3_q = f_RK(X_RK + k1 * (3. * DT_RK / 40.) + k2 * (9. * DT_RK / 40.),
                    U_RK)

    k4, k4_q = f_RK(X_RK + k1 * (3. * DT_RK / 10.) - k2 * (9. * DT_RK / 10.) + k3 * (6. * DT_RK / 5.),
                    U_RK)

    k5, k5_q = f_RK(X_RK + - k1 * (11. * DT_RK / 54.) + k2 * (5. * DT_RK / 2.) - k3 * (70. * DT_RK / 27.)
                    + k4 * (35. * DT_RK / 27.),
                    U_RK)

    k6, k6_q = f_RK(X_RK + k1 * (1631. * DT_RK / 55296.) + k2 * (175. * DT_RK / 512.) + k3 * (575. * DT_RK / 13824.)
                    + k4 * (44275. * DT_RK / 110592.) + k5 * (253. * DT_RK / 4096.),
                    U_RK)

    X_RK = X_RK + DT_RK * (37. * k1 / 378. + 250. * k3 / 621. + 125. * k4 / 594. + 512. * k6 / 1771.)
    Q_RK = Q_RK + DT_RK * (37. * k1 / 378. + 250. * k3 / 621. + 125. * k4 / 594. + 512. * k6 / 1771.)


    return Function('F_RKF45', [X0_RK, U_RK], [X_RK, Q_RK], ['x0', 'p'], ['xf', 'qf'])

def RKF45_time(dae, casadi_type):

    x = dae['x']
    qddot = dae['p']
    xdot = dae['ode']
    L = dae['quad']

    f_RK = Function('f_RK', [x, qddot], [xdot, L])

    nx = x.size1()
    nv = qddot.size1()

    if casadi_type is 'MX':
        X0_RK = MX.sym('X0_RK', nx)
        U_RK = MX.sym('U_RK', nv)
        DT_RK = MX.sym('DT_RK', 1)
    elif casadi_type is 'SX':
        X0_RK = SX.sym('X0_RK', nx)
        U_RK = SX.sym('U_RK', nv)
        DT_RK = SX.sym('DT_RK', 1)
    else:
        raise Exception('Input casadi_type can be only SX or MX!')

    X_RK = X0_RK
    Q_RK = 0

    k1, k1_q = f_RK(X_RK, U_RK)

    k2, k2_q = f_RK(X_RK + k1 * (DT_RK / 5),
                    U_RK)

    k3, k3_q = f_RK(X_RK + k1 * (3. * DT_RK / 40.) + k2 * (9. * DT_RK / 40.),
                    U_RK)

    k4, k4_q = f_RK(X_RK + k1 * (3. * DT_RK / 10.) - k2 * (9. * DT_RK / 10.) + k3 * (6. * DT_RK / 5.),
                    U_RK)

    k5, k5_q = f_RK(X_RK + - k1 * (11. * DT_RK / 54.) + k2 * (5. * DT_RK / 2.) - k3 * (70. * DT_RK / 27.)
                    + k4 * (35. * DT_RK / 27.),
                    U_RK)

    k6, k6_q = f_RK(X_RK + k1 * (1631. * DT_RK / 55296.) + k2 * (175. * DT_RK / 512.) + k3 * (575. * DT_RK / 13824.)
                    + k4 * (44275. * DT_RK / 110592.) + k5 * (253. * DT_RK / 4096.),
                    U_RK)

    X_RK = X_RK + DT_RK * (37. * k1 / 378. + 250. * k3 / 621. + 125. * k4 / 594. + 512. * k6 / 1771.)
    Q_RK = Q_RK + DT_RK * (37. * k1 / 378. + 250. * k3 / 621. + 125. * k4 / 594. + 512. * k6 / 1771.)


    return Function('F_RKF45', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])