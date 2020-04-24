from casadi import *
from horizon import *


def EULER(dae, opts, casadi_type):
    """
    Implements an integration scheme based on Euler integration (http://www.cmth.ph.ic.ac.uk/people/a.mackinnon/Lectures/compphys/node4.html):
        x = x + dt*xdot(x,u)
    Args:
        dae: a dictionary containing
            'x': state
            'p': control
            'ode': a function of the state and control returning the derivative of the state
            'quad': quadrature term
        opts: a dictionary containing 'tf': integration time
        NOTE: this term can be used to take into account also a final time to optimize
        casadi_type: 'SX' or 'MX'

    Returns: Function('F_RK', [X0_RK, U_RK], [X_RK, Q_RK], ['x0', 'p'], ['xf', 'qf'])
        which given in input the actual state X0_RK and control U_RK returns the integrated state X_RK and quadrature term Q_RK

    """
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

    X_RK = X_RK + DT_RK * k1
    Q_RK = Q_RK + DT_RK * k1_q

    return Function('F_RK', [X0_RK, U_RK], [X_RK, Q_RK], ['x0', 'p'], ['xf', 'qf'])

def EULER_time(dae, casadi_type):
    """
        Implements an integration scheme based on Euler integration (http://www.cmth.ph.ic.ac.uk/people/a.mackinnon/Lectures/compphys/node4.html):
            x = x + dt*xdot(x,u)
        Args:
            dae: a dictionary containing
                'x': state
                'p': control (gere including time)
                'ode': a function of the state and control returning the derivative of the state
                'quad': quadrature term
            casadi_type: 'SX' or 'MX'

        Returns: Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])
            which given in input the actual state X0_RK, control U_RK and time DT_RK returns the integrated state X_RK and quadrature term Q_RK

        """
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

    X_RK = X_RK + DT_RK * k1
    Q_RK = Q_RK + DT_RK * k1_q

    return Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])

def RK2(dae, opts, casadi_type):
    """
        Implements an integration scheme based on 2nd-order Runge-Kutta integration (http://www.cmth.ph.ic.ac.uk/people/a.mackinnon/Lectures/compphys/node11.html):
            k1 = xdot(x,u)
            k2 = xdot(x + 0.5*dt*k1, u)
            x = x + dt*k2
        Args:
            dae: a dictionary containing
                'x': state
                'p': control
                'ode': a function of the state and control returning the derivative of the state
                'quad': quadrature term
            opts: a dictionary containing 'tf': integration time
            NOTE: this term can be used to take into account also a final time to optimize
            casadi_type: 'SX' or 'MX'

        Returns: Function('F_RK', [X0_RK, U_RK], [X_RK, Q_RK], ['x0', 'p'], ['xf', 'qf'])
            which given in input the actual state X0_RK and control U_RK returns the integrated state X_RK and quadrature term Q_RK

        """
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
    k2, k2_q = f_RK(X_RK + DT_RK / 2. * k1, U_RK)

    X_RK = X_RK + DT_RK * k2
    Q_RK = Q_RK + DT_RK * k2_q

    return Function('F_RK', [X0_RK, U_RK], [X_RK, Q_RK], ['x0', 'p'], ['xf', 'qf'])

def RK2_time(dae, casadi_type):
    """
    Implements an integration scheme based on 2nd-order Runge-Kutta integration (http://www.cmth.ph.ic.ac.uk/people/a.mackinnon/Lectures/compphys/node11.html):
    k1 = xdot(x,u)
    k2 = xdot(x + 0.5*dt*k1, u)
    x = x + dt*k2
    Args:
        dae: a dictionary containing
            'x': state
            'p': control (gere including time)
            'ode': a function of the state and control returning the derivative of the state
            'quad': quadrature term
        casadi_type: 'SX' or 'MX'

    Returns: Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])
        which given in input the actual state X0_RK, control U_RK and time DT_RK returns the integrated state X_RK and quadrature term Q_RK

    """

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
    k2, k2_q = f_RK(X_RK + DT_RK / 2. * k1, U_RK)

    X_RK = X_RK + DT_RK * k2
    Q_RK = Q_RK + DT_RK * k2_q

    return Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])

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
    k2, k2_q = f_RK(X_RK + DT_RK / 2. * k1, U_RK)
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
    k2, k2_q = f_RK(X_RK + DT_RK / 2. * k1, U_RK)
    k3, k3_q = f_RK(X_RK + DT_RK / 2. * k2, U_RK)
    k4, k4_q = f_RK(X_RK + DT_RK * k3, U_RK)

    X_RK = X_RK + DT_RK / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    Q_RK = Q_RK + DT_RK / 6. * (k1_q + 2. * k2_q + 2. * k3_q + k4_q)

    return Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])

def LEAPFROG(dae, opts, casadi_type):
    x = dae['x']
    qddot = dae['p']
    xdot = dae['ode']
    L = dae['quad']

    f_RK = Function('f_RK', [x, qddot], [xdot, L])

    nx = x.size1()
    nv = qddot.size1()

    if casadi_type is 'MX':
        X0_RK = MX.sym('X0_RK', nx)
        X0_PREV_RK = MX.sym('X0_PREV_RK', nx)
        U_RK = MX.sym('U_RK', nv)
    elif casadi_type is 'SX':
        X0_RK = SX.sym('X0_RK', nx)
        X0_PREV_RK = SX.sym('X0_PREV_RK', nx)
        U_RK = SX.sym('U_RK', nv)
    else:
        raise Exception('Input casadi_type can be only SX or MX!')

    DT_RK = opts['tf']

    Q_RK = 0

    k1, k1_q = f_RK(X0_RK, U_RK)

    X_RK = X0_PREV_RK + 2. * DT_RK * k1
    X_PREV_RK = X0_RK

    return Function('F_RK', [X0_RK, X0_PREV_RK, U_RK], [X_RK, X_PREV_RK, Q_RK], ['x0', 'x0_prev', 'p'],
                    ['xf', 'xf_prev', 'qf'])