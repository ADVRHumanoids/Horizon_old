import sys
import os
sys.path.insert(0, os.path.abspath('../../src'))

import horizon.solvers.ilqr as ilqr
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

# switch between MX and SX
sym_t = cs.MX

# define state
x = sym_t.sym('x', 1)
u = sym_t.sym('u', 1)
xdot = u

N = 10  # number of nodes
dt = 0.1  # discretizaton step
niter = 2  # ilqr iterations
x0 = 0  # initial state (falling)
xf = 1  # desired final state (upright)


l = 0.5*cs.sumsqr(u)/dt  # intermediate cost
lf = cs.sumsqr(x-xf)*2000  # final cost
gf = x - xf

L = cs.Function('intermediate_cost',
                {'x': x, 'u': u, 'l': l},
                ['x', 'u'],
                ['l']
                )

Xdot = cs.Function('dynamics',
                {'x': x, 'u': u, 'xdot': xdot},
                ['x', 'u'],
                ['xdot']
                )


Lf = cs.Function('final_cost',
                {'x': x, 'l': lf},
                ['x'],
                ['l']
                )

Gf = cs.Function('final_constr',
                {'x': x, 'gf': gf},
                ['x'],
                ['gf']
                )


solver = ilqr.IterativeLQR(xdot=Xdot,
                           dt=dt,
                           N=N,
                           diff_intermediate_cost=L,
                           final_cost=Lf,
                           final_constraint=None)

solver.setInitialState(x0)
# # solver.randomizeInitialGuess()
solver.solve(niter)

plt.figure(figsize=[12, 5])
xtrj = np.column_stack(solver._state_trj)
lines = plt.plot(xtrj, 's-')
plt.grid()
plt.show()