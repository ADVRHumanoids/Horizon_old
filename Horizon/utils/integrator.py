from casadi import *
from Horizon.horizon import *


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

    return Function('F_RK', [X0_RK, X0_PREV_RK, U_RK], [X_RK, X_PREV_RK, Q_RK], ['x0', 'x0_prev', 'p'], ['xf', 'xf_prev', 'qf'])

def RKF_time(dae, opts, casadi_type):
    """Runge-Kutta-Fehlberg method based on pseudo-code presented in "Numerical Analysis", 6th Edition,
        by Burden and Faires, Brooks-Cole, 1997."""
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
        DT0_RK = MX.sym('DT0_RK', 1)
    elif casadi_type is 'SX':
        X0_RK = SX.sym('X0_RK', nx)
        U_RK = SX.sym('U_RK', nv)
        DT0_RK = SX.sym('DT0_RK', 1)
    else:
        raise Exception('Input casadi_type can be only SX or MX!')

    X_RK = X0_RK
    DT_RK = DT0_RK
    Q_RK = 0.

    # Coefficients used to compute the dependent variable argument of f

    b21 = 1./4.
    b31 = 3./32.
    b32 = 9./32.
    b41 = 1932./2197.
    b42 = -7200./2197.
    b43 = 7296./2197.
    b51 = 439./216.
    b52 = -8.
    b53 = 3680./513.
    b54 = -845./4104.
    b61 = -8./27.
    b62 = 2.
    b63 = -3544./2565.
    b64 = 1859./4104.
    b65 = -11./40.

    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.

    r1 = 1./360.
    r3 = -128./4275.
    r4 = -2197./75240.
    r5 = 1./50.
    r6 = 2./55.

    # Coefficients used to compute 4th order RK estimate

    c1 = 25./216.
    c3 = 1408./2565.
    c4 = 2197./4104.
    c5 = -1./5.

    # Compute values needed to compute truncation error estimate and
    # the 4th order RK estimate.

    k1, k1_q = f_RK(X_RK, U_RK)
    k1 = DT_RK * k1
    k2, k2_q = f_RK(X_RK + b21 * k1, U_RK)
    k2 = DT_RK * k2
    k3, k3_q = f_RK(X_RK + b31 * k1 + b32 * k2, U_RK)
    k3 = DT_RK * k3
    k4, k4_q = f_RK(X_RK + b41 * k1 + b42 * k2 + b43 * k3, U_RK)
    k4 = DT_RK * k4
    k5, k5_q = f_RK(X_RK + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4, U_RK)
    k5 = DT_RK * k5
    k6, k6_q = f_RK(X_RK + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5, U_RK)
    k6 = DT_RK * k6

    X_RK = X_RK + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5

    # Compute the estimate of the local truncation error
    R = mmax(fabs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6))
    # Now compute next step size
    TOL = opts['tol']
    DT_RK = DT_RK * 0.84 * (TOL * DT_RK / R) ** 0.25

    return Function('F_RK', [X0_RK, U_RK, DT0_RK], [X_RK, DT_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'dtf', 'qf'])
