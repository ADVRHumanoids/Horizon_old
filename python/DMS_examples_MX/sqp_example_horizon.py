#!/usr/bin/env python

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import horizon
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn
import matlogger2.matlogger as matl
import constraints as cons
from utils.resample_integrator import *
from utils.inverse_dynamics import *
from utils.replay_trajectory import *
from utils.integrator import *

from solvers.sqp import *

import matplotlib.pyplot as plt

N = 21  # Control discretization
T = 10.0  # End time

# CREATE VARIABLES
dx, Dx = create_variable('Dx', 2, N, "STATE", "MX")
dx_min = np.array([-inf, -inf]).tolist()
dx_max = np.array([inf, inf]).tolist()
dx_init = np.array([0., 0.]).tolist()

du, Du = create_variable('Du', 1, N, "CONTROL", "MX")
du_min = -1.
du_max = 1.
du_init = 0.0

v_min, v_max = create_bounds({"x_min": [dx_min], "x_max": [dx_max],
                              "u_min": [du_min], "u_max": [du_max]}, N)

print "v_min: ", v_min
print v_min.shape

print "v_max: ",v_max
print v_max.shape


v0 = create_init({"x_init": [dx_init], "u_init": [du_init]}, N)

print "v0: ", v0
print v0.shape

X, U = create_state_and_control([Dx], [Du])

print "X: ", X
print "U: ", U


V = concat_states_and_controls({"X": X, "U": U})
print "V: ", V

# FORMULATE DISCRETE TIME DYNAMICS
# System dynamics
xdot = vertcat(*[(1. - dx[1] ** 2) * dx[0] - dx[1] + du, dx[0]])
dae = {'x': dx, 'p': du, 'ode': xdot, 'quad': []}
opts = {'tf': T/N}
F_integrator = RK4(dae, opts, "MX")

print "F_integrator: ", F_integrator

# INITIAL & FINAL CONDITIONS
v_min[0] = v_max[0] = v0[0] = 0.
v_min[1] = v_max[1] = v0[1] = 1.

v_min[-1] = v_max[-1] = 0.
v_min[-2] = v_max[-2] = 0.
print "v_min: ", v_min
print "v_max: ", v_max

# CONSTRAINTS
G = constraint_handler()

# MULTIPLE SHOOTING CONSTRAINT
integrator_dict = {'x0': X, 'p': U}
multiple_shooting_constraint = multiple_shooting(integrator_dict, F_integrator)
g1, g_min1, g_max1 = constraint(multiple_shooting_constraint, 0, N-1)
G.set_constraint(g1, g_min1, g_max1)

# Gauss-Newton objective
g, g_min, g_max = G.get_constraints()

print "g: ", g
print "g_min: ", g_max
print "g_max: ", g_min

opts = {'max_iter': 10,
        'qpoases.sparse': True,
        'qpoases.linsol_plugin': 'ma57',
        'qpoases.enableRamping': False,
        'qpoases.enableFarBounds': False,
        'qpoases.enableFlippingBounds': False,
        'qpoases.enableFullLITests': False,
        'qpoases.enableNZCTests': False,
        'qpoases.enableDriftCorrection': 0,
        'qpoases.enableCholeskyRefactorisation': 0,
        'qpoases.enableEqualities': True,
        'qpoases.initialStatusBounds': 'inactive',
        'qpoases.numRefinementSteps': 0,
        'qpoases.terminationTolerance': 1e9*np.finfo(float).eps,
        'qpoases.enableInertiaCorrection': False,
        'qpoases.printLevel': 'none',
        'osqp.verbose': False}


t = time.time()
solver = sqp('solver', "osqp", {'f': V, 'x': V, 'g': g}, opts)
solution = solver(x0=v0, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
elapsed = time.time() - t
print "elapsed: ", elapsed

obj_history = solution['f']
print "obj_history: ", obj_history
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
plt.plot(linspace(0, T, N ), x0_opt, '--')
plt.plot(linspace(0, T, N), x1_opt, '-')
plt.step(linspace(0, T, N-1), u_opt, '-.')
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



