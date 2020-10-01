from casadi import *
from numpy import *
import matplotlib.pyplot as plt


# Solve a QP
def qpsolve(H, g, lbx, ubx, A, lba, uba):
    # QP structure
    qp = qpStruct(h=H.sparsity(), a=A.sparsity())

    # Create CasADi solver instance
    if False:
        solver = QpSolver("qpoases", qp)
    else:
        solver = QpSolver("nlp", qp)
        solver.setOption("nlp_solver", "ipopt")

        # Initialize the solver
    solver.init()

    # Pass problem data
    solver.setInput(H, "h")
    solver.setInput(g, "g")
    solver.setInput(A, "a")
    solver.setInput(lbx, "lbx")
    solver.setInput(ubx, "ubx")
    solver.setInput(lba, "lba")
    solver.setInput(uba, "uba")

    # Solver the QP
    solver.evaluate()

    # Return the solution
    return solver.getOutput("x")


N = 20  # Control discretization
T = 10.0  # End time

# Declare variables (use scalar graph)
u = SX.sym("u")  # control
x = SX.sym("x", 2)  # states

# System dynamics
xdot = vertcat(*[(1 - x[1] ** 2) * x[0] - x[1] + u, x[0]])
f = Function('f', [x, u], [xdot])

# RK4 with M steps
U = MX.sym("U")
X = MX.sym("X", 2)
M = 10;
DT = T / (N * M)
XF = X
QF = 0

for j in range(M):
    k1 = f(XF, U)
    k2 = f(XF + DT / 2 * k1, U)
    k3 = f(XF + DT / 2 * k2, U)
    k4 = f(XF + DT * k3, U)
    XF += DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
F = Function('F', [X, U], [XF])

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

# Form function for calculating the Gauss-Newton objective
r_fcn = Function('r_fcn', [v], [r])

# Form function for calculating the constraints
g_fcn = Function('g_fcn', [v], [g])


# Generate functions for the Jacobians
Jac_r_fcn = r_fcn.jacobian()
Jac_g_fcn = g_fcn.jacobian()

# Objective value history
obj_history = []

# Constraint violation history
con_history = []

# Gauss-Newton SQP
v_opt = matrix(v0)
N_iter = 10
for k in range(N_iter):
    # Form quadratic approximation of objective
    Jac_r_fcn.setInput(v_opt)
    Jac_r_fcn.evaluate()
    J_r_k = Jac_r_fcn.getOutput(0)
    r_k = Jac_r_fcn.getOutput(1)

    # Form quadratic approximation of constraints
    Jac_g_fcn.setInput(v_opt)
    Jac_g_fcn.evaluate()
    J_g_k = Jac_g_fcn.getOutput(0)
    g_k = Jac_g_fcn.getOutput(1)

    # Gauss-Newton Hessian
    H_k = mtimes(J_r_k.T, J_r_k)

    # Gradient of the objective function
    Grad_obj_k = mtimes(J_r_k.T, r_k)

    # Bounds on delta_v
    dv_min = vmin - v_opt
    dv_max = vmax - v_opt

    # Solve the QP
    dv = qpsolve(H_k, Grad_obj_k, dv_min, dv_max, J_g_k, -g_k, -g_k)

    # Take the full step
    v_opt += dv

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


