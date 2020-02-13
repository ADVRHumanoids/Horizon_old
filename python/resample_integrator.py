from casadi import *
from horizon import *


def resample_integrator(X, U_integrator, tf, dt, dae):

    ns = np.size(X)
    ti = tf/ns  # interval time

    if dt >= ti:
        dt = ti
        ni = 1
    else:
        ni = int(round(ti / dt))  # number of intermediate nodes in interval

    opts = {'tf': dt}
    F_integrator = integrator('F_integrator', 'rk', dae, opts)

    # Resample X
    n_res = (ns-1)*ni
    nx = X[0].size1()
    X_res = MX(Sparsity.dense(nx, n_res))
    X_res[0:nx, 0] = X[0]

    k = -1

    for i in range(ns-1):  # cycle on intervals
        for j in range(ni):  # cycle on intermediate nodes in interval

            if j == 0:
                 X_res[0:nx, k+1] = F_integrator(x0=X[i], p=U_integrator[i])['xf']
            else:
                 X_res[0:nx, k+1] = F_integrator(x0=X_res[0:nx, k], p=U_integrator[i])['xf']

            k += 1

    return X_res

