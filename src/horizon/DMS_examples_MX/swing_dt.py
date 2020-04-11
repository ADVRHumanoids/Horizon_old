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

logger = matl.MatLogger2('/tmp/swing_dt_log')
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
ns = 60  # number of shooting nodes

nc = 3  # number of contacts

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)

DoF = nq - 7  # Contacts + anchor_rope + rope

nv = kindyn.nv()  # Velocity DoFs

nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

# CREATE VARIABLES
dt, Dt = create_variable('Dt', 1, ns, "CONTROL", "MX")
dt_min = 0.02
dt_max = 0.03
dt_init = 0.01

t_final = ns*dt_min

q, Q = create_variable("Q", nq, ns, "STATE", "MX")

q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                  -0.3, -0.1, -0.1,  # Contact 1
                  -0.3, -0.05, -0.1,  # Contact 2
                  -1.57, -1.57, -3.1415,  # rope_anchor
                  0.3]).tolist()  # rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0,  # Floating base
                  0.3, 0.05, 0.1,  # Contact 1
                  0.3, 0.1, 0.1,  # Contact 2
                  1.57, 1.57, 3.1415,  # rope_anchor
                  0.3]).tolist()  # rope
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0., 0., 0.,
                   0., 0., 0.,
                   0., 1., 0.,
                   0.3]).tolist()

qdot, Qdot = create_variable('Qdot', nv, ns, "STATE", "MX")
qdot_min = (-100.*np.ones(nv)).tolist()
qdot_max = (100.*np.ones(nv)).tolist()
qdot_init = np.zeros(nv).tolist()

qddot, Qddot = create_variable('Qddot', nv, ns, "CONTROL", "MX")
qddot_min = (-100.*np.ones(nv)).tolist()
qddot_max = (100.*np.ones(nv)).tolist()
qddot_init = np.zeros(nv).tolist()
qddot_init[2] = -9.8

f1, F1 = create_variable('F1', nf, ns, "CONTROL", "MX")
f_min1 = (-10000.*np.ones(nf)).tolist()
f_max1 = (10000.*np.ones(nf)).tolist()
f_init1 = np.zeros(nf).tolist()

f2, F2 = create_variable('F2', nf, ns, "CONTROL", "MX")
f_min2 = (-10000.*np.ones(nf)).tolist()
f_max2 = (10000.*np.ones(nf)).tolist()
f_init2 = np.zeros(nf).tolist()

fRope, FRope = create_variable('FRope', nf, ns, "CONTROL", "MX")
f_minRope = (-10000.*np.ones(nf)).tolist()
f_maxRope = (10000.*np.ones(nf)).tolist()
f_initRope = np.zeros(nf).tolist()

x, xdot = dynamic_model_with_floating_base(q, qdot, qddot)

L = 0.5*dot(qdot, qdot)  # Objective term

# FORMULATE DISCRETE TIME DYNAMICS
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
F_integrator = RKF45_time(dae, "MX")

# START WITH AN EMPTY NLP
X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, FRope, Dt])
V = concat_states_and_controls({"X": X, "U": U})
v_min, v_max = create_bounds({"x_min": [q_min, qdot_min], "x_max": [q_max, qdot_max],
                              "u_min": [qddot_min, f_min1, f_min2, f_minRope, dt_min], "u_max": [qddot_max, f_max1, f_max2, f_maxRope, dt_max]}, ns)

# SET UP COST FUNCTION
J = MX([0])

q_rest = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0.,
                   0.3]).tolist()


K = 30000.
min_q = lambda k: K*dot(Q[k][7:-1]-q_rest[7:-1], Q[k][7:-1]-q_rest[7:-1])
J += cost_function(min_q, 0, ns)

D = 1.
min_qdot_legs = lambda k: D*dot(Qdot[k][6:12], Qdot[k][6:12])
J += cost_function(min_qdot_legs, 0, ns)

min_qdot = lambda k: 1.*dot(Qdot[k][6:-1], Qdot[k][6:-1])
J += cost_function(min_qdot, 0, ns)

min_qddot_a = lambda k: 1.*dot(Qddot[k][6:-1], Qddot[k][6:-1])
J += cost_function(min_qddot_a, 0, ns-1)


# min_F1 = lambda k: 1000.*dot(F1[k], F1[k])
# J += cost_function(min_F1, 0, ns-1)

# min_F2 = lambda k: 1000.*dot(F2[k], F2[k])
# J += cost_function(min_F2, 0, ns-1)

# min_FRope = lambda k: 1.*dot(FRope[k], FRope[k])
# J += cost_function(min_FRope, 0, ns-1)

min_deltaFRope = lambda k: 1.*dot(FRope[k]-FRope[k-1], FRope[k]-FRope[k-1])  # min Fdot
J += cost_function(min_deltaFRope, 1, ns-1)

# min_deltaDt = lambda k: 1.*(Dt[k]-Dt[k-1])
# J += cost_function(min_deltaDt, 1, ns-1)
#
# min_deltaQddot = lambda k: 1.*dot(Qddot[k]-Qddot[k-1], Qddot[k]-Qddot[k-1])  # min Fdot
# J += cost_function(min_deltaQddot, 1, ns-1)

# min_Tf = lambda k: 1000.*Tf[0]
# J += cost_function(min_Tf, 0, ns-1)

# min_Dt = lambda k: 100000.*(Dt[k]-dt_max)
# J += cost_function(min_Dt, 0, ns-1)

# CONSTRAINTS
G = constraint_handler()

# INITIAL CONDITION CONSTRAINT
x_init = q_init + qdot_init
init = cons.initial_condition.initial_condition(X[0], x_init)
g1, g_min1, g_max1 = constraint(init, 0, 1)
G.set_constraint(g1, g_min1, g_max1)

# MULTIPLE SHOOTING CONSTRAINT
integrator_dict = {'x0': X, 'p': Qddot, 'time': Dt}
multiple_shooting_constraint = multiple_shooting(integrator_dict, F_integrator)

g2, g_min2, g_max2 = constraint(multiple_shooting_constraint, 0, ns-1)
G.set_constraint(g2, g_min2, g_max2)

# INVERSE DYNAMICS CONSTRAINT
dd = {'rope_anchor2': FRope}
id = inverse_dynamics(Q, Qdot, Qddot, ID, dd, kindyn)

tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    -1000., -1000., -1000.,  # Contact 1
                    -1000., -1000., -1000.,  # Contact 2
                    0., 0., 0.,  # rope_anchor
                    -10000.]).tolist()  # rope

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

# MINIMUM FINAL TIME
g6, g_min6, g_max6 = cons.final_time.final_time(Dt, t_final)
G.set_constraint(g6, g_min6, g_max6)

opts = {'ipopt.tol': 1e-3,
        'ipopt.constr_viol_tol': 1e-4,
        'ipopt.max_iter': 3000,
        'ipopt.linear_solver': 'ma57'}

g, g_min, g_max = G.get_constraints()
solver = nlpsol('solver', 'ipopt', {'f': J, 'x': V, 'g': g}, opts)

x0 = create_init({"x_init": [q_init, qdot_init], "u_init": [qddot_init, f_init1, f_init2, f_initRope, dt_init]}, ns)

sol = solver(x0=x0, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
w_opt = sol['x'].full().flatten()

# RETRIEVE SOLUTION AND LOGGING
solution_dict = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'FRope': FRope, 'Dt': Dt}, w_opt)
q_hist = solution_dict['Q']
dt_hist = solution_dict['Dt']

tf = 0.0

for i in range(ns-1):
    tf += dt_hist[i]

# RESAMPLE STATE FOR REPLAY TRAJECTORY
dt = 0.001
X_res = resample_integrator(X, Qddot, dt_hist, dt, dae)
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
logger.add('Tf', tf)

del(logger)

# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

replay_trajectory(dt, joint_list, q_hist_res).replay()