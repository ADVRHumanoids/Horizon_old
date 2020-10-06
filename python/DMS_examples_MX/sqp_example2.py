from casadi import *
from numpy import *
import matplotlib.pyplot as plt

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from solvers.sqp import *


N = 20  # Control discretization
T = 10.0  # End time

# Declare variables (use scalar graph)
u = SX.sym("u")  # control
x = SX.sym("x", 2)  # states

# System dynamics
xdot = vertcat(*[(1 - x[1] ** 2) * x[0] - x[1] + u, x[0]])
f = Function('f', {'x':x, 'u':u, 'xdot':xdot}, ['x', 'u'], ['xdot'])

# RK4 with M steps
U = MX.sym("U")
X = MX.sym("X", 2)
M = 10
DT = T / (N * M)
XF = X
QF = 0

for j in range(M):
    k1 = f(XF, U)
    k2 = f(XF + DT / 2 * k1, U)
    k3 = f(XF + DT / 2 * k2, U)
    k4 = f(XF + DT * k3, U)
    XF += DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
F = Function('F', {'X': X, 'U': U, 'XF': XF}, ['X', 'U'], ['XF'])

# Formulate NLP (use matrix graph)
nv = 1 * N + 2 * (N + 1)
v = MX.sym("v", nv)

# Get the state for each shooting interval
xk = [v[3 * k: 3 * k + 2] for k in range(N + 1)]

# Get the control for each shooting interval
uk = [v[3 * k + 2] for k in range(N)]

# Variable bounds
vmin = -inf * ones(nv)
vmax = inf * ones(nv)

# Initial solution guess
v0 = zeros(nv)

# Control bounds
vmin[2::3] = -1.0
vmax[2::3] = 1.0

# Initial condition
vmin[0] = vmax[0] = v0[0] = 0
vmin[1] = vmax[1] = v0[1] = 1

# Terminal constraint
vmin[-2] = vmax[-2] = v0[-2] = 0
vmin[-1] = vmax[-1] = v0[-1] = 0

# Constraint function with bounds
g = []
gmin = []
gmax = []

# Build up a graph of integrator calls
for k in range(N):
    # Call the integrator
    xf = F(xk[k], uk[k])

    # Append continuity constraints
    g.append(xf - xk[k + 1])
    gmin.append(zeros(2))
    gmax.append(zeros(2))

# Concatenate constraints
g = vertcat(*g)
gmin = concatenate(gmin)
gmax = concatenate(gmax)

# Gauss-Newton objective
r = v

opts = {'max_iter': 10}
solver = sqp('solver', {'f': r, 'x': v, 'g': g}, opts)
solution = solver(x0=v0, lbx=vmin, ubx=vmax, lbg=gmin, ubg=gmax)

obj_history = solution['f']
con_history = solution['g']
v_opt = solution['x']

# Print result
print "solution found: ", v_opt

# Retrieve the solution
x0_opt = v_opt[0::3]
x1_opt = v_opt[1::3]
u_opt = v_opt[2::3]

# Plot the results
plt.figure(1)
plt.clf()
plt.subplot(121)
plt.plot(linspace(0, T, N + 1), x0_opt, '--')
plt.plot(linspace(0, T, N + 1), x1_opt, '-')
plt.step(linspace(0, T, N), u_opt, '-.')
plt.title("Solution: Gauss-Newton SQP")
plt.xlabel('time')
plt.legend(['x0 trajectory', 'x1 trajectory', 'u trajectory'])
plt.grid()

plt.subplot(122)
plt.title("SQP solver output")
plt.semilogy(obj_history)
plt.semilogy(con_history)
plt.xlabel('iteration')
plt.legend(['Objective value', 'Constraint violation'], loc='center right')
plt.grid()

plt.show()


