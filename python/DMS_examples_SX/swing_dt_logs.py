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
from utils.conversions_to_euler import *

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
ns = 70  # number of shooting nodes

nc = 3  # number of contacts

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)

DoF = nq - 7  # Contacts + anchor_rope + rope

nv = kindyn.nv()  # Velocity DoFs

nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

# CREATE VARIABLES
dt, Dt = create_variable('Dt', 1, ns, "CONTROL")
dt_min = 0.01
dt_max = 0.01
dt_init = dt_min

t_final = ns*dt_min

q, Q = create_variable("Q", nq, ns, "STATE")

q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                  0., 0., 0.,  # Contact 1
                  0., -0.,0.,  # Contact 2
                  -1.57, -1.57, -3.1415,  # rope_anchor
                  0.3]).tolist()  # rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0,  # Floating base
                  0., 0., 0.,  # Contact 1
                  0., 0., 0.,  # Contact 2
                  1.57, 1.57, 3.1415,  # rope_anchor
                  0.3]).tolist()  # rope
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0., 0., 0.,
                   0., 0., 0.,
                   0., 1., 0.,
                   0.3]).tolist()

qdot, Qdot = create_variable('Qdot', nv, ns, "STATE")
qdot_min = (-1000.*np.ones(nv)).tolist()
qdot_max = (1000.*np.ones(nv)).tolist()
qdot_init = np.zeros(nv).tolist()

qddot, Qddot = create_variable('Qddot', nv, ns, "CONTROL")
qddot_min = (-1000.*np.ones(nv)).tolist()
qddot_max = (1000.*np.ones(nv)).tolist()
qddot_init = np.zeros(nv).tolist()
qddot_init[2] = -9.8

f1, F1 = create_variable('F1', nf, ns, "CONTROL")
f_min1 = (0.*np.ones(nf)).tolist()
f_max1 = (0.*np.ones(nf)).tolist()
f_init1 = np.zeros(nf).tolist()

f2, F2 = create_variable('F2', nf, ns, "CONTROL")
f_min2 = (0.*np.ones(nf)).tolist()
f_max2 = (0.*np.ones(nf)).tolist()
f_init2 = np.zeros(nf).tolist()

fRope, FRope = create_variable('FRope', nf, ns, "CONTROL")
f_minRope = (-10000.*np.ones(nf)).tolist()
f_maxRope = (10000.*np.ones(nf)).tolist()
f_initRope = np.zeros(nf).tolist()

x, xdot = dynamic_model_with_floating_base(q, qdot, qddot)

L = 0.5*dot(qdot, qdot)  # Objective term

# FORMULATE DISCRETE TIME DYNAMICS
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
F_integrator = RK4_time(dae, "SX")

# START WITH AN EMPTY NLP
X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, FRope, Dt])
V = concat_states_and_controls({"X": X, "U": U})
v_min, v_max = create_bounds({"x_min": [q_min, qdot_min], "x_max": [q_max, qdot_max],
                              "u_min": [qddot_min, f_min1, f_min2, f_minRope, dt_min], "u_max": [qddot_max, f_max1, f_max2, f_maxRope, dt_max]}, ns)

# SET UP COST FUNCTION
J = SX([0])

min_X = lambda k: 100.*dot(Qdot[k], Qdot[k])
J += cost_function(min_X, 0, ns)

min_U = lambda k: 1.*dot(U[k], U[k])
J += cost_function(min_U, 0, ns-1)

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

# # # MINIMUM FINAL TIME
# g6, g_min6, g_max6 = cons.final_time.final_time(Dt, t_final)
# G.set_constraint(g6, g_min6, g_max6)

opts = {'ipopt.tol': 1e-3,
        'ipopt.constr_viol_tol': 1e-3,
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
dt_hist = solution_dict['Dt']

tf = 0.0

for i in range(ns-1):
    tf += dt_hist[i]


# GET ADDITIONAL VARIABLES
Tau = id.compute_nodes(0, ns-1)
get_Tau = Function("get_Tau", [V], [Tau], ['V'], ['Tau'])
tau_hist = (get_Tau(V=w_opt)['Tau'].full().flatten()).reshape(ns-1, nv)

# LOGGING
for k in solution_dict:
    logger.add(k, solution_dict[k])

FKcomputer = kinematics(kindyn, Q, Qdot, Qddot)

# Contact1_pos = FKcomputer.computeFK('Contact1', 'ee_pos', 0, ns)
# get_Contact1_pos = Function("get_Contact1_pos", [V], [Contact1_pos], ['V'], ['Contact1_pos'])
# Contact1_pos_hist = (get_Contact1_pos(V=w_opt)['Contact1_pos'].full().flatten()).reshape(ns, 3)
#
# Contact2_pos = FKcomputer.computeFK('Contact2', 'ee_pos', 0, ns)
# get_Contact2_pos = Function("get_Contact2_pos", [V], [Contact2_pos], ['V'], ['Contact2_pos'])
# Contact2_pos_hist = (get_Contact2_pos(V=w_opt)['Contact2_pos'].full().flatten()).reshape(ns, 3)
#
# Waist_pos = FKcomputer.computeFK('Waist', 'ee_pos', 0, ns)
# get_Waist_pos = Function("get_Waist_pos", [V], [Waist_pos], ['V'], ['Waist_pos'])
# Waist_pos_hist = (get_Waist_pos(V=w_opt)['Waist_pos'].full().flatten()).reshape(ns, 3)
#
# Waist_rot = FKcomputer.computeFK('Waist', 'ee_rot', 0, ns)
# get_Waist_rot = Function("get_Waist_rot", [V], [Waist_rot], ['V'], ['Waist_rot'])
# Waist_rot_hist = (get_Waist_rot(V=w_opt)['Waist_rot'].full().flatten()).reshape(ns, 3, 3)
# # CONVERSION TO EULER ANGLES
# Waist_rot_hist = rotation_matrix_to_euler(Waist_rot_hist)
#
#

MasterPoint_pos = FKcomputer.computeFK('rope_anchor1_3', 'ee_pos', 0, ns)
get_MasterPoint_pos = Function("get_MasterPoint_pos", [V], [MasterPoint_pos], ['V'], ['MasterPoint_pos'])
MasterPoint_pos_hist = (get_MasterPoint_pos(V=w_opt)['MasterPoint_pos'].full().flatten()).reshape(ns, 3)

MasterPoint_rot = FKcomputer.computeFK('rope_anchor1_3', 'ee_rot', 0, ns)
get_MasterPoint_rot = Function("get_MasterPoint_rot", [V], [MasterPoint_rot], ['V'], ['MasterPoint_rot'])
MasterPoint_rot_hist = (get_MasterPoint_rot(V=w_opt)['MasterPoint_rot'].full().flatten()).reshape(ns, 3, 3)
# # CONVERSION TO EULER ANGLES
MasterPoint_rot_hist = rotation_matrix_to_euler(MasterPoint_rot_hist)

MasterPoint_vel_linear = FKcomputer.computeFK('rope_anchor1_3', 'ee_vel_linear', 0, ns)
get_MasterPoint_vel_linear = Function("get_MasterPoint_vel_linear", [V], [MasterPoint_vel_linear], ['V'], ['MasterPoint_vel_linear'])
MasterPoint_vel_linear_hist = (get_MasterPoint_vel_linear(V=w_opt)['MasterPoint_vel_linear'].full().flatten()).reshape(ns, 3)

MasterPoint_vel_angular = FKcomputer.computeFK('rope_anchor1_3', 'ee_vel_angular', 0, ns)
get_MasterPoint_vel_angular = Function("get_MasterPoint_vel_angular", [V], [MasterPoint_vel_angular], ['V'], ['MasterPoint_vel_angular'])
MasterPoint_vel_angular_hist = (get_MasterPoint_vel_angular(V=w_opt)['MasterPoint_vel_angular'].full().flatten()).reshape(ns, 3)

BaseLink_pos = FKcomputer.computeFK('base_link', 'ee_pos', 0, ns)
get_BaseLink_pos = Function("get_BaseLink_pos", [V], [BaseLink_pos], ['V'], ['BaseLink_pos'])
BaseLink_pos_hist = (get_BaseLink_pos(V=w_opt)['BaseLink_pos'].full().flatten()).reshape(ns, 3)

BaseLink_vel_linear = FKcomputer.computeFK('base_link', 'ee_vel_linear', 0, ns)
get_BaseLink_vel_linear = Function("get_BaseLink_vel_linear", [V], [BaseLink_vel_linear], ['V'], ['BaseLink_vel_linear'])
BaseLink_vel_linear_hist = (get_BaseLink_vel_linear(V=w_opt)['BaseLink_vel_linear'].full().flatten()).reshape(ns, 3)

BaseLink_vel_angular = FKcomputer.computeFK('base_link', 'ee_vel_angular', 0, ns)
get_BaseLink_vel_angular = Function("get_BaseLink_vel_angular", [V], [BaseLink_vel_angular], ['V'], ['BaseLink_vel_angular'])
BaseLink_vel_angular_hist = (get_BaseLink_vel_angular(V=w_opt)['BaseLink_vel_angular'].full().flatten()).reshape(ns, 3)

#FloatingBase_J = FKcomputer.computeFK('base_link','ee_jacobian', 0, ns)
#get_FloatingBase_J = Function("get_FloatingBase_J", [V], [FloatingBase_J], ['V'], ['FloatingBase_J'])
#FloatingBase_J_hist = (get_FloatingBase_J(V=w_opt)['FloatingBase_J'].full().flatten()).reshape(ns, 6, nv)

# MasterPoint_acc_linear = FKcomputer.computeFK('rope_anchor1_3', 'ee_acc_linear', 0, ns-1)
# get_MasterPoint_acc_linear = Function("get_MasterPoint_acc_linear", [V], [MasterPoint_acc_linear], ['V'], ['MasterPoint_acc_linear'])
# MasterPoint_acc_linear_hist = (get_MasterPoint_acc_linear(V=w_opt)['MasterPoint_acc_linear'].full().flatten()).reshape(ns-1, 3)
#
# AnchorPoint_pos = FKcomputer.computeFK('rope_anchor2', 'ee_pos', 0, ns)
# get_AnchorPoint_pos = Function("get_AnchorPoint_pos", [V], [AnchorPoint_pos], ['V'], ['AnchorPoint_pos'])
# AnchorPoint_pos_hist = (get_AnchorPoint_pos(V=w_opt)['AnchorPoint_pos'].full().flatten()).reshape(ns, 3)
#
# CoM_pos = FKcomputer.computeCoM('com', 0, ns)
# get_CoM_pos = Function("get_CoM_pos", [V], [CoM_pos], ['V'], ['CoM_pos'])
# CoM_pos_hist = (get_CoM_pos(V=w_opt)['CoM_pos'].full().flatten()).reshape(ns, 3)
#
# CoM_vel = FKcomputer.computeCoM('vcom', 0, ns)
# get_CoM_vel = Function("get_CoM_vel", [V], [CoM_vel], ['V'], ['CoM_vel'])
# CoM_vel_hist = (get_CoM_vel(V=w_opt)['CoM_vel'].full().flatten()).reshape(ns, 3)
#
# CoM_acc = FKcomputer.computeCoM('acom', 0, ns-1)
# get_CoM_acc = Function("get_CoM_acc", [V], [CoM_acc], ['V'], ['CoM_acc'])
# CoM_acc_hist = (get_CoM_acc(V=w_opt)['CoM_acc'].full().flatten()).reshape(ns-1, 3)
#



logger.add('Tau', tau_hist)
logger.add('Tf', tf)
logger.add('w_opt', w_opt)
# logger.add('Contact1', Contact1_pos_hist)
# logger.add('Contact2', Contact2_pos_hist)
# logger.add('Waist_pos', Waist_pos_hist)
# logger.add('Waist_rot', Waist_rot_hist)
# logger.add('fb_pos', q_hist[:, 0:3])
# logger.add('fb_rot', quaternion_to_euler(q_hist[:, 3:7]))
logger.add('MasterPoint', MasterPoint_pos_hist)
logger.add('MasterPoint_rot', MasterPoint_rot_hist)
logger.add('MasterPoint_vel_lin', MasterPoint_vel_linear_hist)
logger.add('MasterPoint_vel_ang', MasterPoint_vel_angular_hist)
# logger.add('MasterPoint_acc', MasterPoint_acc_linear_hist)
# logger.add('AnchorPoint', AnchorPoint_pos_hist)
# logger.add('CoM_pos', CoM_pos_hist)
# logger.add('CoM_vel', CoM_vel_hist)
# logger.add('CoM_acc', CoM_acc_hist)
# logger.add('BaseLink', BaseLink_pos_hist)
logger.add('BaseLink', BaseLink_pos_hist)
logger.add('BaseLink_vel_lin', BaseLink_vel_linear_hist)
logger.add('BaseLink_vel_ang', BaseLink_vel_angular_hist)
#logger.add('FloatingBase_J', FloatingBase_J_hist)




# RESAMPLE STATE FOR REPLAY TRAJECTORY
dt = 0.001
X_res = resample_integrator(X, Qddot, dt_hist, dt, dae)
get_X_res = Function("get_X_res", [V], [X_res], ['V'], ['X_res'])
x_hist_res = get_X_res(V=w_opt)['X_res'].full()
q_hist_res = (x_hist_res[0:nq, :]).transpose()

logger.add('Q_res', q_hist_res)

del(logger)

# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

replay_trajectory(dt, joint_list, q_hist_res).replay()