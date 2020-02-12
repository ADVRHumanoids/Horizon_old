from casadi import *
from horizon import *


def resample_integrator(X, Qddot, tf, dt, dae):

    opts = {'tf': dt}
    F_integrator = integrator('F_integrator', 'rk', dae, opts)

    ns = np.size(X)
    ti = tf/ns  # interval time
    ni = int(round(ti/dt))  # number of intermediate nodes in interval

    # Resample X
    n_res = ns*ni
    nx = X[0].size1()
    X_res = MX(Sparsity.dense(nx, n_res))
    X_res[0:nx, 0] = X[0]

    k = 0

    for i in range(ns-1):  # cycle on intervals
        for j in range(ni):  # cycle on intermediate nodes in interval

            if j == 0:
                 X_res[0:nx, k+1] = F_integrator(x0=X[i], p=Qddot[i])['xf']
            else:
                 X_res[0:nx, k+1] = F_integrator(x0=X_res[0:nx, k], p=Qddot[i])['xf']

            k += 1

        # print k

    for i in range(n_res-k):
        X_res[0:nx, k+i] = X[-1]

    # Resample Qddot
    ns = np.size(Qddot)
    n_res = ns*ni
    nv = Qddot[0].size1()
    Qddot_res = MX(Sparsity.dense(nv, n_res))
    Qddot_res[0:nv, 0] = Qddot[0]

    k = 0
    for i in range(ns-2):  # cycle on intervals
        for j in range(ni):  # cycle on intermediate nodes in interval
            Qddot_res[0:nv, k + 1] = Qddot[i]
            k += 1

    for i in range((n_res-k)):
        Qddot_res[0:nv, k+i] = Qddot[-1]


    return X_res, Qddot_res


def normalize(v):
    return v/np.linalg.norm(v)
