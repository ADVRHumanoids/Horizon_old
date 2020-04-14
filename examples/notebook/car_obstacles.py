#!/usr/bin/env python
# coding: utf-8

# # Inverted pendulum on cart model
# 
# Equations:
# 
# $$
# \begin{cases}
# \ddot{x} + \alpha\cos\theta\,\ddot{\theta} + \alpha\sin\theta\,\dot{\theta}^2 &= u \\
# -\cos\theta\,\ddot{x} + \ddot{\theta} &= \sin\theta
# \end{cases},
# $$
# 
# where $\alpha < 1$ is the ratio of the pendulum mass over the total system mass.
# 
# We now define the system dynamics in CasADi

# In[18]:

import sys
import os
sys.path.insert(0, os.path.abspath('../../src'))


import horizon.solvers.ilqr as ilqr
import casadi as cs
import numpy as np

# switch between MX and SX
sym_t = cs.MX

# define state
x = sym_t.sym('x', 3)
u = sym_t.sym('u', 2)
xdot = cs.vertcat( u[0] * cs.cos(x[2]), 
                   u[0] * cs.sin(x[2]),
                   u[1])


# ## Optimal control problem
# 
# We optimize the following cost
# $$
#     J(\boldsymbol{x}_0, \mathbf{U}) = \int_0^{t_f} u^2 \,\text{d}t \; + \; 1000\cdot\lVert \boldsymbol{x}_N - \boldsymbol{x}_f^\text{ref}  \rVert^2
# $$
# 
# discretized with $N$ knots of duration $\text{d}t$.

# In[19]:


N = 50  # number of nodes
dt = 0.1  # discretizaton step
niter = 50  # ilqr iterations
x0 = np.array([0, 0, 0])  # initial state (falling)
xf = np.array([0, 1, 0])  # desired final state (upright)

obs_center = np.array([0.05, 0.5])
obs_r = 0.1
obs = -cs.sumsqr(x[0:1] - obs_center) + obs_r  # obs(x) < 0


def barrier(x):
    
    return 0.5*(x + 0.001*cs.sqrt((x*1000)**2 + 1e-9))
    

l = cs.sumsqr(u)  # intermediate cost
lf = 100*cs.sumsqr(x - xf)  # final cost
gf = x - xf


# Costs and dynamics must be expressed as CasADi functions

# In[20]:


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
np.random.seed(11311)
solver._use_single_shooting_state_update = True
# solver._use_second_order_dynamics = True
solver.randomizeInitialGuess()
solver.solve(0)


if False:

    import matplotlib.pyplot as plt

    plt.figure(figsize=[12, 5])
    xtrj = np.column_stack(solver._state_trj)
    lines = plt.plot(xtrj[0,:], xtrj[1,:], 's-')
    circle = plt.Circle(obs_center, radius=obs_r, fc='r')
    # plt.gca().add_patch(circle)
    plt.title('XY trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    # plt.legend(lines, ['x', 'y', r'$\theta$'])


    # In[23]:
    plt.figure(figsize=[12, 5])
    plt.plot(solver._dcost, label='cost')
    plt.plot(solver._dx_norm, label='dx')
    plt.plot(solver._du_norm, label='du')
    plt.title('Increments')
    plt.xlabel('Iteration')
    plt.semilogy()
    plt.grid()
    plt.legend()

    plt.figure(figsize=[12, 5])
    lines = plt.plot(solver._state_trj)
    plt.title('State trajectory')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.grid()
    plt.legend(lines, ['x', 'y', r'$\theta$'])

    plt.figure(figsize=[12, 5])
    lines = plt.plot(solver._ctrl_trj)
    plt.title('Control trajectory')
    plt.xlabel('Time')
    plt.ylabel('Control')
    plt.grid()
    plt.legend(lines, ['v', r'$\dot{\theta}$'])

    plt.figure(figsize=[12, 5])
    lines = plt.plot(solver._defect)
    plt.title('Dynamics error')
    plt.xlabel('Time')
    plt.ylabel('State defect')
    plt.grid()
    plt.legend(lines, ['x', 'y', r'$\theta$'])

print(solver._dcost)

cost_est = 0.0

for i in range(len(solver._ctrl_trj)):
    cost_est += dt* np.linalg.norm(solver._ctrl_trj[i])**2

cost_est += Lf(x=solver._state_trj[-1])['l'].__float__()
