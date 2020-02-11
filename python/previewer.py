from casadi import *
from horizon import *


def resample_solution(X, Qddot, tf, dt, dae):
    ns = np.size(X)
    nx = X[0].size1()

    ti = tf/ns #interval time

    ni = int(round(ti/dt)) # number of intermidiate nodes in interval

    n_res = ns*ni



    X_res = MX(Sparsity.dense(nx, n_res))

    opts = {'tf': dt}
    F_integrator = integrator('F_integrator', 'rk', dae, opts)

    X_res[0:nx, 0] = X[0]

    k = 0
    for i in range(ns-1): # cycle on intervals
        for j in range(ni): # cycle on intermediate nodes in interval

            if j == 0:
                 X_res[0:nx, k+1] = F_integrator(x0=X[i], p=Qddot[i])['xf']
            else:
                 X_res[0:nx, k+1] = F_integrator(x0=X_res[0:nx, k], p=Qddot[i])['xf']

            k += 1

        print k
    for i in range(n_res-k):
        X_res[0:nx, k+i] = X[-1]

    return X_res


def normalize(v):
    return v/np.linalg.norm(v)