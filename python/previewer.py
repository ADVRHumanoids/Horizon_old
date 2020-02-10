from casadi import *

def trajectory_resampler(ns, integrator, V, X, Qddot, tf, dt, nq, nx, w_opt):
    n_res = int(round(tf/dt))

    q_res = MX(Sparsity.dense(nq, n_res))
    X_res = MX(Sparsity.dense(nx, n_res + 1))

    k = 0
    for i in range(ns-1):
        for j in range(int(round(n_res/ns))):

            if j == 0:
                X_res[0:nx,k+1] = integrator(x0=X[i], p=Qddot[i])['xf']
            else:
                X_res[0:nx,k+1] = integrator(x0=X_res[k], p=Qddot[i])['xf']

            q_res[0:nq,k] = X_res[0:nq, k + 1]

            k += 1

    resampler = Function("Resampler", [V], [q_res], ['V'], ['q_res'])

    return resampler(V=w_opt)['q_res'].full()

def normalize(v):
    return v/np.linalg.norm(v)