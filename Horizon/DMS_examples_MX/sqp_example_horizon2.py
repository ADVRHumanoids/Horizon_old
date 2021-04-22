#!/usr/bin/env python

import Horizon.horizon
import Horizon.constraints as cons
from Horizon.utils.resample_integrator import *
from Horizon.utils.inverse_dynamics import *
from Horizon.utils.replay_trajectory import *
from Horizon.utils.integrator import *

from Horizon.solvers import sqp

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

print ("v_min: ", v_min)
print (v_min.shape)

print ("v_max: ",v_max)
print (v_max.shape)


v0 = create_init({"x_init": [dx_init], "u_init": [du_init]}, N)

print ("v0: ", v0)
print (v0.shape)

X, U = create_state_and_control([Dx], [Du])

print ("X: ", X)
print ("U: ", U)


V = concat_states_and_controls({"X": X, "U": U})
print ("V: ", V)

# FORMULATE DISCRETE TIME DYNAMICS
# System dynamics
xdot = vertcat(*[(1. - dx[1] ** 2) * dx[0] - dx[1] + du, dx[0]])
dae = {'x': dx, 'p': du, 'ode': xdot, 'quad': []}
opts = {'tf': T/N}
F_integrator = RK4(dae, opts, "MX")
print('xdot: ', xdot)

print ("F_integrator: ", F_integrator)

# INITIAL & FINAL CONDITIONS
v_min[0] = v_max[0] = v0[0] = 0.
v_min[1] = v_max[1] = v0[1] = 1.

v_min[-1] = v_max[-1] = 0.
v_min[-2] = v_max[-2] = 0.
print ("v_min: ", v_min)
print ("v_max: ", v_max)

# CONSTRAINTS
G = constraint_handler()

# MULTIPLE SHOOTING CONSTRAINT
integrator_dict = {'x0': X, 'p': U}
multiple_shooting_constraint = multiple_shooting(integrator_dict, F_integrator)
g1, g_min1, g_max1 = constraint(multiple_shooting_constraint, 0, N-1)
G.set_constraint(g1, g_min1, g_max1)

# Gauss-Newton objective
g, g_min, g_max = G.get_constraints()

print ("g: ", g)
print ("g_min: ", g_min)
print ("g_max: ", g_max)

ordered_G = ordered_constraint_handler()
ordered_G.set_constraint(multiple_shooting_constraint, 0, N-1)
ord_g, ord_g_min, ord_g_max = ordered_G.get_constraints()

print ("ord_g: ", ord_g)
print ("ord_g_min: ", ord_g_min)
print ("ord_g_max: ", ord_g_max)

GG = constraint_handler()
GG.set_constraint(ord_g, ord_g_min, ord_g_max)
gg, gg_min, gg_max = G.get_constraints()

print ("gg: ", gg)
print ("gg_min: ", gg_min)
print ("gg_max: ", gg_max)


d = {'verbose': False}
opts = {'max_iter': 10,
        'osqp.osqp': d}


print 'V: ', V


minV = lambda k: vertcat(X[k], U[k])
FF = ordered_cost_function_handler()
FF.set_cost_function(minV, 0, N-1)
FF.set_cost_function(lambda k: X[-1], N-1, N)
JJ = FF.get_cost_function()
print 'JJ: ', JJ


t = time.time()
solver = sqp.sqp('solver', 'qpoases', {'f': vertcat(*JJ), 'x': V, 'g': g}, opts)

solution = solver(x0=v0, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
#solver = sqp('solver', "osqp", {'f': V, 'x': V}, opts)
#solution = solver(x0=v0, lbx=v_min, ubx=v_max)
elapsed = time.time() - t
print ("first solve: ", elapsed)

print ("compute Hessian time: ", solver.get_hessian_computation_time())
print ("compute QP time: ", solver.get_qp_computation_time())

obj_history = solution['f']
print ("obj_history: ", obj_history)
con_history = solution['g']
print ("con_history: ", con_history)
v_opt = solution['x']

# Print result
print ("solution found: ", v_opt)



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

Wx = 10.
minV = lambda k: vertcat(Wx*X[k], U[k])
FF = ordered_cost_function_handler()
FF.set_cost_function(minV, 0, N-1)
FF.set_cost_function(lambda k: Wx*X[-1], N-1, N)
JJ = FF.get_cost_function()
print 'JJ: ', JJ


tic = time.time()
solver.f(vertcat(*JJ))
toc = time.time()
print ("cost function update time: ", toc-tic)

t = time.time()
solution = solver(x0=v0, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
#solver = sqp('solver', "osqp", {'f': V, 'x': V}, opts)
#solution = solver(x0=v0, lbx=v_min, ubx=v_max)
elapsed = time.time() - t
print ("second solve: ", elapsed)

print ("compute Hessian time: ", solver.get_hessian_computation_time())
print ("compute QP time: ", solver.get_qp_computation_time())

obj_history = solution['f']
print ("obj_history: ", obj_history)
con_history = solution['g']
print ("con_history: ", con_history)
v_opt = solution['x']

# Print result
print ("solution found: ", v_opt)



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



