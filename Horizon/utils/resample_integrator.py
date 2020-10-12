from casadi import *
from Horizon.horizon import *
from Horizon.utils.integrator import *
from Horizon.utils.inverse_dynamics import *

def resample_integrator(X, U_integrator, time, dt, dae, ID, dict, kindyn, force_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL):
    """
    Resample solution to a different number of nodes, RK4 integrator is used for the resampling
    Args:
        X: state variable
        U_integrator: control variable
        time: the final time (tf) or the vector of intermediate periods (dt_hist)
        dt: resampled period
        dae: a dictionary containing
                'x': state
                'p': control
                'ode': a function of the state and control returning the derivative of the state
                'quad': quadrature term
        ID: Function.deserialize(kindyn.rnea()) TODO: remove, this can be taken from kindyn
        dict: dictionary containing a map between frames and force variables e.g. {'lsole': F1}
        kindyn: object of type casadi_kin_dyn
        force_reference_frame: this is the frame which is used to compute the Jacobian during the ID computation:
                LOCAL (default)
                WORLD
                LOCAL_WORLD_ALIGNED

    Returns:
        X_res: resampled state
        Tau_res: resampled torques
        TODO: add resampled controls!
    """

    ns = np.size(X)
    nx = X[0].size1()
    nv = U_integrator[0].size1()
    nq = nv + 1

    X_res = []
    Tau_res = []

    for key in dict:
        Jac = Function.deserialize(kindyn.jacobian(key, force_reference_frame))

    if np.size(time) == 1:

        ti = time / ns  # interval time

        if dt >= ti:
            dt = ti
            ni = 1
        else:
            ni = int(round(ti / dt))  # number of intermediate nodes in interval

        opts = {'tf': dt}
        if type(X[0]) is casadi.SX:
            F_integrator = RK4(dae, opts, 'SX')
        elif type(X[0]) is casadi.MX:
            F_integrator = RK4(dae, opts, 'MX')
        else:
            raise Exception('Input type can be only casadi.SX or casadi.MX!')

        # Resample X
        n_res = (ns - 1) * ni
        if type(X[0]) is casadi.SX:
            X_res = SX(Sparsity.dense(nx, n_res))
            Tau_res = SX(Sparsity.dense(nv, n_res))
        elif type(X[0]) is casadi.MX:
            X_res = MX(Sparsity.dense(nx, n_res))
            Tau_res = MX(Sparsity.dense(nv, n_res))

        X_res[0:nx, 0] = X[0]

        k = -1

        for i in range(ns - 1):  # cycle on intervals
            for j in range(ni):  # cycle on intermediate nodes in interval

                if j == 0:
                    X_res[0:nx, k + 1] = F_integrator(x0=X[i], p=U_integrator[i])['xf']
                else:
                    X_res[0:nx, k + 1] = F_integrator(x0=X_res[0:nx, k], p=U_integrator[i])['xf']

                if type(X[0]) is casadi.SX:
                    JtF_k = SX([0])
                    zeros = SX.zeros(3, 1)
                elif type(X[0]) is casadi.MX:
                    JtF_k = MX([0])
                    zeros = MX.zeros(3, 1)

                for key in dict:
                    Jac_k = Jac(q=X_res[0:nq, k + 1])['J']
                    Force_k = dict[key][i]
                    JtF = mtimes(Jac_k.T, vertcat(Force_k, zeros))
                    JtF_k = JtF_k + JtF

                Tau_res[0:nv, k + 1] = ID(q=X_res[0:nq, k + 1], v=X_res[nq:nx, k + 1], a=U_integrator[i])['tau'] - JtF_k

                k += 1

    else:

        opts = {'tf': dt}
        if type(X[0]) is casadi.SX:
            F_integrator = RK4(dae, opts, 'SX')
        elif type(X[0]) is casadi.MX:
            F_integrator = RK4(dae, opts, 'MX')
        else:
            raise Exception('Input type can be only casadi.SX or casadi.MX!')

        ni = {}
        n_res = 0
        for i in range(np.size(time)):
            ni[i] = int(round(time[i] / dt))
            n_res += ni[i]

        # Resample X
        if type(X[0]) is casadi.SX:
            X_res = SX(Sparsity.dense(nx, n_res))
            Tau_res = SX(Sparsity.dense(nv, n_res))
        elif type(X[0]) is casadi.MX:
            X_res = MX(Sparsity.dense(nx, n_res))
            Tau_res = MX(Sparsity.dense(nv, n_res))

        k = -1

        for i in range(ns - 1):  # cycle on intervals
            for j in range(ni[i]):  # cycle on intermediate nodes in interval

                if j == 0:
                    X_res[0:nx, k + 1] = F_integrator(x0=X[i], p=U_integrator[i])['xf']
                else:
                    X_res[0:nx, k + 1] = F_integrator(x0=X_res[0:nx, k], p=U_integrator[i])['xf']

                if type(X[0]) is casadi.SX:
                    JtF_k = SX([0])
                    zeros = SX.zeros(3, 1)
                elif type(X[0]) is casadi.MX:
                    JtF_k = MX([0])
                    zeros = MX.zeros(3, 1)

                for key in dict:
                    Jac_k = Jac(q=X_res[0:nq, k + 1])['J']
                    Force_k = dict[key][i]
                    JtF = mtimes(Jac_k.T, vertcat(Force_k, zeros))
                    JtF_k = JtF_k + JtF

                Tau_res[0:nv, k + 1] = ID(q=X_res[0:nq, k + 1], v=X_res[nq:nx, k + 1], a=U_integrator[i])['tau'] - JtF_k

                k += 1

    return X_res, Tau_res

def resample_integrator_with_controls(X, U, U_integrator, time, dt, dae, ID, dict, kindyn, force_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL):
    """
    Resample solution to a different number of nodes, RK4 integrator is used for the resampling
    Args:
        X: state variable
        U_integrator: control variable
        time: the final time (tf) or the vector of intermediate periods (dt_hist)
        dt: resampled period
        dae: a dictionary containing
                'x': state
                'p': control
                'ode': a function of the state and control returning the derivative of the state
                'quad': quadrature term
        ID: Function.deserialize(kindyn.rnea()) TODO: remove, this can be taken from kindyn
        dict: dictionary containing a map between frames and force variables e.g. {'lsole': F1}
        kindyn: object of type casadi_kin_dyn
        force_reference_frame: this is the frame which is used to compute the Jacobian during the ID computation:
                LOCAL (default)
                WORLD
                LOCAL_WORLD_ALIGNED

    Returns:
        X_res: resampled state
        Tau_res: resampled torques
        TODO: add resampled controls!
    """

    ns = np.size(X)
    nx = X[0].size1()
    nu = U[0].size1()
    nv = U_integrator[0].size1()
    nq = nv + 1

    X_res = []
    Tau_res = []
    U_res = []

    for key in dict:
        Jac = Function.deserialize(kindyn.jacobian(key, force_reference_frame))

    if np.size(time) == 1:

        ti = time / ns  # interval time

        if dt >= ti:
            dt = ti
            ni = 1
        else:
            ni = int(round(ti / dt))  # number of intermediate nodes in interval

        opts = {'tf': dt}
        if type(X[0]) is casadi.SX:
            F_integrator = RK4(dae, opts, 'SX')
        elif type(X[0]) is casadi.MX:
            F_integrator = RK4(dae, opts, 'MX')
        else:
            raise Exception('Input type can be only casadi.SX or casadi.MX!')

        # Resample X
        n_res = (ns - 1) * ni
        if type(X[0]) is casadi.SX:
            X_res = SX(Sparsity.dense(nx, n_res))
            U_res = SX(Sparsity.dense(nx, n_res))
            Tau_res = SX(Sparsity.dense(nv, n_res))
        elif type(X[0]) is casadi.MX:
            X_res = MX(Sparsity.dense(nx, n_res))
            U_res = MX(Sparsity.dense(nx, n_res))
            Tau_res = MX(Sparsity.dense(nv, n_res))

        X_res[0:nx, 0] = X[0]
        U_res[0:nu, 0] = U[0]

        k = -1

        for i in range(ns - 1):  # cycle on intervals
            for j in range(ni):  # cycle on intermediate nodes in interval

                if j == 0:
                    X_res[0:nx, k + 1] = F_integrator(x0=X[i], p=U_integrator[i])['xf']
                else:
                    X_res[0:nx, k + 1] = F_integrator(x0=X_res[0:nx, k], p=U_integrator[i])['xf']

                U_res[0:nu, k + 1] = U[i]

                if type(X[0]) is casadi.SX:
                    JtF_k = SX([0])
                    zeros = SX.zeros(3, 1)
                elif type(X[0]) is casadi.MX:
                    JtF_k = MX([0])
                    zeros = MX.zeros(3, 1)

                for key in dict:
                    Jac_k = Jac(q=X_res[0:nq, k + 1])['J']
                    Force_k = dict[key][i]
                    JtF = mtimes(Jac_k.T, vertcat(Force_k, zeros))
                    JtF_k = JtF_k + JtF

                Tau_res[0:nv, k + 1] = ID(q=X_res[0:nq, k + 1], v=X_res[nq:nx, k + 1], a=U_integrator[i])['tau'] - JtF_k

                k += 1

    else:

        opts = {'tf': dt}
        if type(X[0]) is casadi.SX:
            F_integrator = RK4(dae, opts, 'SX')
        elif type(X[0]) is casadi.MX:
            F_integrator = RK4(dae, opts, 'MX')
        else:
            raise Exception('Input type can be only casadi.SX or casadi.MX!')

        ni = {}
        n_res = 0
        for i in range(np.size(time)):
            ni[i] = int(round(time[i] / dt))
            n_res += ni[i]

        # Resample X
        if type(X[0]) is casadi.SX:
            X_res = SX(Sparsity.dense(nx, n_res))
            U_res = SX(Sparsity.dense(nu, n_res))
            Tau_res = SX(Sparsity.dense(nv, n_res))
        elif type(X[0]) is casadi.MX:
            X_res = MX(Sparsity.dense(nx, n_res))
            U_res = MX(Sparsity.dense(nu, n_res))
            Tau_res = MX(Sparsity.dense(nv, n_res))

        k = -1

        for i in range(ns - 1):  # cycle on intervals
            for j in range(ni[i]):  # cycle on intermediate nodes in interval

                if j == 0:
                    X_res[0:nx, k + 1] = F_integrator(x0=X[i], p=U_integrator[i])['xf']
                else:
                    X_res[0:nx, k + 1] = F_integrator(x0=X_res[0:nx, k], p=U_integrator[i])['xf']

                U_res[0:nu, k+1] = U[i]

                if type(X[0]) is casadi.SX:
                    JtF_k = SX([0])
                    zeros = SX.zeros(3, 1)
                elif type(X[0]) is casadi.MX:
                    JtF_k = MX([0])
                    zeros = MX.zeros(3, 1)

                for key in dict:
                    Jac_k = Jac(q=X_res[0:nq, k + 1])['J']
                    Force_k = dict[key][i]
                    JtF = mtimes(Jac_k.T, vertcat(Force_k, zeros))
                    JtF_k = JtF_k + JtF

                Tau_res[0:nv, k + 1] = ID(q=X_res[0:nq, k + 1], v=X_res[nq:nx, k + 1], a=U_integrator[i])['tau'] - JtF_k

                k += 1

    return X_res, U_res, Tau_res
