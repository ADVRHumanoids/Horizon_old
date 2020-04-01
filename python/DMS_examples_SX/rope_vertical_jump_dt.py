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
from utils.kinematics import *
from utils.normalize_quaternion import *
from utils.rotation_matrix_to_euler import *

logger = matl.MatLogger2('/tmp/rope_vertical_jump_dt_log')
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
ns = 70  # number of shooting nodes

nc = 3  # number of contacts

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)

DoF = nq - 7  # Contacts + anchor_rope + rope

nv = kindyn.nv()  # Velocity DoFs

nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

# CREATE VARIABLES
dt, Dt = create_variable('Dt', 1, ns, 'CONTROL', 'SX')
dt_min = 0.01
dt_max = 0.08
dt_init = dt_min

t_final = ns*dt_min

q, Q = create_variable('Q', nq, ns, 'STATE', 'SX')

foot_z_offset = 0.#0.5

jump_length = 0.6

q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                  -0.3, -0.1, -0.1+foot_z_offset,  # Contact 1
                  -0.3, -0.05, -0.1+foot_z_offset,  # Contact 2
                  -1.57, -1.57, -3.1415,  # rope_anchor
                  0.3]).tolist()  # rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0,  # Floating base
                  0.3, 0.05, 0.1+foot_z_offset,  # Contact 1
                  0.3, 0.1, 0.1+foot_z_offset,  # Contact 2
                  1.57, 1.57, 3.1415,  # rope_anchor
                  0.3+jump_length]).tolist()  # rope

# STATE BEFORE JUMP
alpha = 0.3
rope_init_lenght = 0.3
x_foot = rope_init_lenght * np.sin(alpha)
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   x_foot, 0., 0.+foot_z_offset,
                   x_foot, 0., 0.+foot_z_offset,
                   0., alpha, 0.,
                   rope_init_lenght]).tolist()
print "q_init: ", q_init

# STATE AFTER VERTICAL JUMP
q_final = np.array([0.0, 0.0, -jump_length, 0.0, 0.0, 0.0, 1.0,
                   x_foot, 0., 0.+foot_z_offset,
                   x_foot, 0., 0.+foot_z_offset,
                   0., alpha, 0.,
                   rope_init_lenght+jump_length]).tolist()
print "q_final: ", q_final

qdot, Qdot = create_variable('Qdot', nv, ns, 'STATE', 'SX')
qdot_min = (-100.*np.ones(nv)).tolist()
qdot_max = (100.*np.ones(nv)).tolist()
qdot_init = np.zeros(nv).tolist()

qddot, Qddot = create_variable('Qddot', nv, ns, 'CONTROL', 'SX')
qddot_min = (-100.*np.ones(nv)).tolist()
qddot_max = (100.*np.ones(nv)).tolist()
qddot_init = np.zeros(nv).tolist()
qddot_init[2] = -9.8

f1, F1 = create_variable('F1', nf, ns, 'CONTROL', 'SX')
f_min1 = (-10000.*np.ones(nf)).tolist()
f_max1 = (10000.*np.ones(nf)).tolist()
f_init1 = np.zeros(nf).tolist()

f2, F2 = create_variable('F2', nf, ns, 'CONTROL', 'SX')
f_min2 = (-10000.*np.ones(nf)).tolist()
f_max2 = (10000.*np.ones(nf)).tolist()
f_init2 = np.zeros(nf).tolist()

fRope, FRope = create_variable('FRope', nf, ns, 'CONTROL', 'SX')
f_minRope = (-10000.*np.ones(nf)).tolist()
f_maxRope = (10000.*np.ones(nf)).tolist()
f_initRope = np.zeros(nf).tolist()

x, xdot = dynamic_model_with_floating_base(q, qdot, qddot)

L = 0.5*dot(qdot, qdot)  # Objective term

# FORMULATE DISCRETE TIME DYNAMICS
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
F_integrator = RKF45_time(dae, 'SX')

# START WITH AN EMPTY NLP
X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, FRope, Dt])
V = concat_states_and_controls({"X": X, "U": U})
v_min, v_max = create_bounds({"x_min": [q_min, qdot_min], "x_max": [q_max, qdot_max],
                              "u_min": [qddot_min, f_min1, f_min2, f_minRope, dt_min], "u_max": [qddot_max, f_max1, f_max2, f_maxRope, dt_max]}, ns)

lift_node = 2 #20
touch_down_node = 60

# SET UP COST FUNCTION
J = SX([0])

q_trg = np.array([-.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                  0.0, 0.0, 0.0+foot_z_offset,
                  0.0, 0.0, 0.0+foot_z_offset,
                  0.0, 0.0, 0.0,
                  0.3]).tolist()

K = 6.5*1e5
min_qd = lambda k: K*dot(Q[k][0]-q_trg[0], Q[k][0]-q_trg[0])
J += cost_function(min_qd, lift_node+1, touch_down_node)

min_qd2 = lambda k: K*dot(Q[k][3:7]-q_trg[3:7], Q[k][3:7]-q_trg[3:7])
J += cost_function(min_qd2, lift_node+1, touch_down_node)

min_qdot = lambda k: 100.*dot(Qdot[k][0:-1], Qdot[k][0:-1])
J += cost_function(min_qdot, 0, ns)

min_qddot = lambda k: 20.*dot(Qddot[k][0:-1], Qddot[k][0:-1])
J += cost_function(min_qddot, 0, ns-1)

min_q = lambda k: K*dot(Q[k], Q[k])
J += cost_function(min_q, touch_down_node, ns)

#min_FC = lambda k: 1.*dot(F1[k]+F2[k], F1[k]+F2[k])
#J += cost_functionSX(min_FC, 0, ns-1)

# min_deltaFC = lambda k: 1*dot((F1[k]-F1[k-1])+(F2[k]-F2[k-1]), (F1[k]-F1[k-1])+(F2[k]-F2[k-1])) # min Fdot
# J += cost_functionSX(min_deltaFC, 1, ns-1)


#min_deltaFRope = lambda k: 1.*dot(FRope[k]-FRope[k-1], FRope[k]-FRope[k-1])  # min Fdot
#J += cost_function(min_deltaFRope, 1, ns-1)

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
# dd = {'rope_anchor2': FRope}
dd = {'rope_anchor2': FRope, 'Contact1': F1, 'Contact2': F2}
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

# WALL
mu = 0.5

R_wall = np.zeros([3, 3])

# R_wall[0, 1] = -1.0
# R_wall[1, 2] = -1.0
# R_wall[2, 0] = 1.0

R_wall[0, 2] = 1.0
R_wall[1, 1] = 1.0
R_wall[2, 0] = -1.0

# STANCE PHASE
contact_handler_F1 = cons.contact.contact_handler(FKR, F1)
contact_handler_F1.setContactAndFrictionCone(Q, q_init, mu, R_wall)
g, g_min, g_max = constraint(contact_handler_F1, 0, lift_node+1)
G.set_constraint(g, g_min, g_max)

contact_handler_F2 = cons.contact.contact_handler(FKL, F2)
contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_wall)
g, g_min, g_max = constraint(contact_handler_F2, 0, lift_node+1)
G.set_constraint(g, g_min, g_max)

# FLIGHT PHASE
contact_handler_F1.removeContact()
g, g_min, g_max = constraint(contact_handler_F1, lift_node+1, touch_down_node)
G.set_constraint(g, g_min, g_max)

contact_handler_F2.removeContact()
g, g_min, g_max = constraint(contact_handler_F2, lift_node+1, touch_down_node)
G.set_constraint(g, g_min, g_max)

# TOUCH DOWN
contact_handler_F1.setContactAndFrictionCone(Q, q_final, mu, R_wall)
g, g_min, g_max = constraint(contact_handler_F1, touch_down_node, ns)
G.set_constraint(g, g_min, g_max)

contact_handler_F2.setContactAndFrictionCone(Q, q_final, mu, R_wall)
g, g_min, g_max = constraint(contact_handler_F2, touch_down_node, ns)
G.set_constraint(g, g_min, g_max)


opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 4000,
        'ipopt.linear_solver': 'ma57'}

g, g_min, g_max = G.get_constraints()
solver = nlpsol('solver', 'ipopt', {'f': J, 'x': V, 'g': g}, opts)

x0 = create_init({"x_init": [q_init, qdot_init], "u_init": [qddot_init, f_init1, f_init2, f_initRope, dt_init]}, ns)

sol = solver(x0=x0, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
w_opt = sol['x'].full().flatten()

# RETRIEVE SOLUTION AND LOGGING
solution_dict = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'FRope': FRope, 'Dt': Dt}, w_opt)

q_hist = solution_dict['Q']
q_hist = normalize_quaternion(q_hist)

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
# NORMALIZE QUATERNION
q_hist_res = normalize_quaternion(q_hist_res)

# GET ADDITIONAL VARIABLES
Tau = id.compute_nodes(0, ns-1)
get_Tau = Function("get_Tau", [V], [Tau], ['V'], ['Tau'])
tau_hist = (get_Tau(V=w_opt)['Tau'].full().flatten()).reshape(ns-1, nv)

# LOGGING
for k in solution_dict:
    logger.add(k, solution_dict[k])

FKcomputer = kinematics(kindyn, Q)
Contact1_pos = FKcomputer.computeFK('Contact1', 'ee_pos', 0, ns)
get_Contact1_pos = Function("get_Contact1_pos", [V], [Contact1_pos], ['V'], ['Contact1_pos'])
Contact1_pos_hist = (get_Contact1_pos(V=w_opt)['Contact1_pos'].full().flatten()).reshape(ns, 3)

Contact2_pos = FKcomputer.computeFK('Contact2', 'ee_pos', 0, ns)
get_Contact2_pos = Function("get_Contact2_pos", [V], [Contact2_pos], ['V'], ['Contact2_pos'])
Contact2_pos_hist = (get_Contact2_pos(V=w_opt)['Contact2_pos'].full().flatten()).reshape(ns, 3)

Waist_pos = FKcomputer.computeFK('Waist', 'ee_pos', 0, ns)
get_Waist_pos = Function("get_Waist_pos", [V], [Waist_pos], ['V'], ['Waist_pos'])
Waist_pos_hist = (get_Waist_pos(V=w_opt)['Waist_pos'].full().flatten()).reshape(ns, 3)

Waist_rot = FKcomputer.computeFK('Waist', 'ee_rot', 0, ns)
get_Waist_rot = Function("get_Waist_rot", [V], [Waist_rot], ['V'], ['Waist_rot'])
Waist_rot_hist = (get_Waist_rot(V=w_opt)['Waist_rot'].full().flatten()).reshape(ns, 3, 3)
# CONVERSION TO EULER ANGLES
Waist_rot_hist = rotation_matrix_to_euler(Waist_rot_hist)


logger.add('Q_res', q_hist_res)
logger.add('Tau', tau_hist)
logger.add('Tf', tf)
logger.add('Contact1', Contact1_pos_hist)
logger.add('Contact2', Contact2_pos_hist)
logger.add('Waist_pos', Waist_pos_hist)
logger.add('Waist_rot', Waist_rot_hist)

del(logger)

# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

replay_trajectory(dt, joint_list, q_hist_res).replay()
