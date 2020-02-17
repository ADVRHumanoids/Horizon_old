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

logger = matl.MatLogger2('/tmp/template_rope_log')
logger.setBufferMode(matl.BufferMode.CircularBuffer)

urdf = rospy.get_param('robot_description')
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# Forward Kinematics of interested links
FK_waist = Function.deserialize(kindyn.fk('Waist'))
FKR = Function.deserialize(kindyn.fk('Contact1'))
FKL = Function.deserialize(kindyn.fk('Contact2'))
FKRope = Function.deserialize(kindyn.fk('rope_anchor2'))

# Inverse Dynamics
ID = Function.deserialize(kindyn.rnea())

# Jacobians
Jac_waist = Function.deserialize(kindyn.jacobian('Waist'))
Jac_CRope = Function.deserialize(kindyn.jacobian('rope_anchor2'))

# OPTIMIZATION PARAMETERS
ns = 30  # number of shooting nodes

nc = 3  # number of contacts

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)

DoF = nq - 7  # Contacts + anchor_rope + rope

nv = kindyn.nv()  # Velocity DoFs

nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

# CREATE VARIABLES
tf, Tf = create_variable("Tf", 1, 1, "FINAL_STATE")
tf_min = 0.0
tf_max = 10.0
tf_init = 1.0

q, Q = create_variable("Q", nq, ns, "STATE")

q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                  -0.3, -0.1, -0.1,  # Contact 1
                  -0.3, -0.05, -0.1,  # Contact 2
                  -1.57, -1.57, -3.1415,  # rope_anchor
                  0.0]).tolist()  # rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0,  # Floating base
                  0.3, 0.05, 0.1,  # Contact 1
                  0.3, 0.1, 0.1,  # Contact 2
                  1.57, 1.57, 3.1415,  # rope_anchor
                  10.0]).tolist()  # rope
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0.,
                   0.1]).tolist()


qdot, Qdot = create_variable('Qdot', nv, ns, "STATE")
qdot_min = (-100.*np.ones(nv)).tolist()
qdot_max = (100.*np.ones(nv)).tolist()
qdot_init = np.zeros(nv).tolist()

qddot, Qddot = create_variable('Qddot', nv, ns, "CONTROL")
qddot_min = (-100.*np.ones(nv)).tolist()
qddot_max = (100.*np.ones(nv)).tolist()
qddot_init = np.zeros(nv).tolist()
qddot_init[2] = -9.8

f1, F1 = create_variable('F1', nf, ns, "CONTROL")
f_min1 = (-10000.*np.ones(nf)).tolist()
f_max1 = (10000.*np.ones(nf)).tolist()
f_init1 = np.zeros(nf).tolist()

f2, F2 = create_variable('F2', nf, ns, "CONTROL")
f_min2 = (-10000.*np.ones(nf)).tolist()
f_max2 = (10000.*np.ones(nf)).tolist()
f_init2 = np.zeros(nf).tolist()

fRope, FRope = create_variable('FRope', nf, ns, "CONTROL")
f_minRope = (-10000.*np.ones(nf)).tolist()
f_maxRope = (10000.*np.ones(nf)).tolist()
f_initRope = np.zeros(nf).tolist()

x, xdot = dynamic_model_with_floating_base(q, qdot, qddot)

L = 0.5*dot(qdot, qdot)  # Objective term

# Runge-Kutta 4 integrator with time
f_RK = Function('f_RK', [x, qddot], [xdot, L])
X0_RK = MX.sym('X0_RK', nq+nv)
U_RK = MX.sym('U_RK', nv)
DT_RK = MX.sym('DT_RK', 1)
X_RK = X0_RK
Q_RK = 0

k1, k1_q = f_RK(X_RK, U_RK)
k2, k2_q = f_RK(X_RK + 0.5*DT_RK*k1, U_RK)
k3, k3_q = f_RK(X_RK + DT_RK / 2 * k2, U_RK)
k4, k4_q = f_RK(X_RK + DT_RK * k3, U_RK)
X_RK = X_RK + DT_RK / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
Q_RK = Q_RK + DT_RK / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)

F_integrator_time = Function('F_RK', [X0_RK, U_RK, DT_RK], [X_RK, Q_RK], ['x0', 'p', 'time'], ['xf', 'qf'])

# START WITH AN EMPTY NLP
X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, FRope, Tf])
V = concat_states_and_controls(X, U)
v_min, v_max = create_bounds([q_min, qdot_min], [q_max, qdot_max], [qddot_min, f_min1, f_min2, f_minRope], [qddot_max, f_max1, f_max2, f_maxRope], ns)

exit()

# SET UP COST FUNCTION
J = MX([0])

min_qdot = lambda k: 100.*dot(Qdot[k][6:-1], Qdot[k][6:-1])
J += cost_function(min_qdot, 0, ns)

min_qddot_a = lambda k: 1000.*dot(Qddot[k][6:-1], Qddot[k][6:-1])
J += cost_function(min_qddot_a, 0, ns-1)

min_F1 = lambda k: 1000.*dot(F1[k], F1[k])
J += cost_function(min_F1, 0, ns-1)

min_F2 = lambda k: 1000.*dot(F2[k], F2[k])
J += cost_function(min_F2, 0, ns-1)

min_FRope = lambda k: 1000.*dot(FRope[k]-FRope[k-1], FRope[k]-FRope[k-1])  # min Fdot
J += cost_function(min_FRope, 1, ns-1)

# CONSTRAINTS
G = constraint_handler()

# INITIAL CONDITION CONSTRAINT
x_init = q_init + qdot_init
init = cons.initial_condition.initial_condition(X[0], x_init)
g1, g_min1, g_max1 = constraint(init, 0, 1)
G.set_constraint(g1, g_min1, g_max1)

# MULTIPLE SHOOTING CONSTRAINT
multiple_shooting_constraint = multiple_shooting(X, Qddot, F_integrator)
g2, g_min2, g_max2 = constraint(multiple_shooting_constraint, 0, ns-1)
G.set_constraint(g2, g_min2, g_max2)

# INVERSE DYNAMICS CONSTRAINT
dd = {'rope_anchor2': FRope}
id = inverse_dynamics(Q, Qdot, Qddot, ID, dd, kindyn)

tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    -1000., -1000., -1000.,  # Contact 1
                    -1000., -1000., -1000.,  # Contact 2
                    0., 0., 0.,  # rope_anchor
                    0.]).tolist()  # rope

tau_max = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    1000., 1000., 1000.,  # Contact 1
                    1000., 1000., 1000.,  # Contact 2
                    0., 0., 0.,  # rope_anchor
                    0.0]).tolist()  # rope

torque_lims1 = cons.torque_limits.torque_lims(id, tau_min, tau_max)
g3, g_min3, g_max3 = constraint(torque_lims1, 0, ns-1)
G.set_constraint(g3, g_min3, g_max3)

# ROPE CONTACT CONSTRAINT
contact_constr = cons.contact.contact(FKRope, Q, q_init)
g5, g_min5, g_max5 = constraint(contact_constr, 0, ns)
G.set_constraint(g5, g_min5, g_max5)


opts = {'ipopt.tol': 1e-4,
        'ipopt.max_iter': 2000,
        'ipopt.linear_solver': 'ma57'}

g, g_min, g_max = G.get_constraints()
solver = nlpsol('solver', 'ipopt', {'f': J, 'x': V, 'g': g}, opts)

x0 = create_init([q_init, qdot_init], [qddot_init, f_init1, f_init2, f_initRope], ns)


sol = solver(x0=x0, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
w_opt = sol['x'].full().flatten()


# RETRIEVE SOLUTION AND LOGGING
solution_dict = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'FRope': FRope}, w_opt)
q_hist = solution_dict['Q']

# RESAMPLE STATE FOR REPLAY TRAJECTORY
dt = 0.001
X_res = resample_integrator(X, Qddot, tf, dt, dae)
get_X_res = Function("get_X_res", [V], [X_res], ['V'], ['X_res'])
x_hist_res = get_X_res(V=w_opt)['X_res'].full()
q_hist_res = (x_hist_res[0:nq, :]).transpose()

# GET ADDITIONAL VARIABLES
Tau = id.compute_nodes(0, ns-1)
get_Tau = Function("get_Tau", [V], [Tau], ['V'], ['Tau'])
tau_hist = (get_Tau(V=w_opt)['Tau'].full().flatten()).reshape(ns-1, nv)

# LOGGING
for k in solution_dict:
    logger.add(k, solution_dict[k])

logger.add('Q_res', q_hist_res)
logger.add('Tau', tau_hist)

del(logger)

# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

replay_trajectory(dt, joint_list, q_hist_res).replay()