import casadi as cs
import pkg_resources
from Horizon.solvers import nilqr
import casadi as cs
import numpy as np



casadi_version = pkg_resources.get_distribution('casadi').version
print("CASADI MAJOR: ", casadi_version[0])
print("CASADI MINOR: ", casadi_version[2])

sym_t = cs.MX

x = sym_t.sym('x', 3)
u = sym_t.sym('u', 2)

f = cs.sumsqr(x) + cs.sumsqr(u)

print ("f: ", f)

Ff = cs.Function('my_func', {'x': x, 'u': u, 'f': f}, ['x', 'u'], ['f'])

print ("Ff: ", Ff)

Jac_Ff = Ff.jac()

print ("Jac_Ff: ", Jac_Ff)

print("Jac_Ff_DfDx: ", Jac_Ff(x=[1,1,1], u=[1,1])["DfDx"])
print("Jac_Ff_DfDu: ", Jac_Ff(x=[1,1,1], u=[1,1])["DfDu"])

Jfx = cs.jacobian(f,x)
FJfx = cs.Function('my_func2', {'x': x, 'u': u, 'Jfx': Jfx}, ['x', 'u'], ['Jfx'])
print("Jfx: ", FJfx(x=[1,1,1], u=[1,1]))

Jfu = cs.jacobian(f,u)
FJfu = cs.Function('my_func3', {'x': x, 'u': u, 'Jfu': Jfu}, ['x', 'u'], ['Jfu'])
print("Jfu: ", FJfu(x=[1,1,1], u=[1,1]))


sym_t = cs.SX

x = sym_t.sym('x', 2)
u = sym_t.sym('u', 2)

xdot = u
N = 5  # number of nodes
dt = 0.01  # discretizaton step
niter = 1  # ilqr iterations
x0 = np.array([0, 0])
xf = np.array([1, 1])

l = cs.sumsqr(u) + cs.sumsqr(x)   # intermediate cost
lf = 200*cs.sumsqr(x - xf)  # final cost
gf = x - xf

constr = {'constr': x[0]}

solver = nilqr.nIterativeLQR(x = x, u = u, xdot=xdot,
                           dt=dt, N=N,
                           intermediate_cost=l,
                           final_cost=lf,
                           intermediate_constraints = constr,
                           final_constraint=x[0]
                             )

solver.setInitialState(x0)
np.random.seed(11311)
solver._use_single_shooting_state_update = True
# solver._use_second_order_dynamics = True
solver.randomizeInitialGuess()
solver.solve(niter)

if True:

    import matplotlib.pyplot as plt

    plt.figure(figsize=[12, 5])
    xtrj = np.column_stack(solver._state_trj)
    lines = plt.plot(xtrj[0,:], xtrj[1,:], 's-')
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
    plt.legend(lines, ['x', 'y'])

    plt.figure(figsize=[12, 5])
    lines = plt.plot(solver._ctrl_trj)
    plt.title('Control trajectory')
    plt.xlabel('Time')
    plt.ylabel('Control')
    plt.grid()
    plt.legend(lines, ['v'])

    plt.figure(figsize=[12, 5])
    lines = plt.plot(solver._defect)
    plt.title('Dynamics error')
    plt.xlabel('Time')
    plt.ylabel('State defect')
    plt.grid()
    plt.legend(lines, ['x', 'y'])

    plt.show()

print(solver._dcost)

