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
import matplotlib.pyplot as plt
import decimal
from utils.footsteps_scheduler import *
import collections
from utils.normalize_quaternion import *
from utils.conversions_to_euler import *
from utils.dt_RKF import *

logger = []
logger = matl.MatLogger2('/tmp/rope_walking_log')
logger.setBufferMode(matl.BufferMode.CircularBuffer)

urdf = rospy.get_param('robot_description')
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

FKR = Function.deserialize(kindyn.fk('Contact1'))
FKL = Function.deserialize(kindyn.fk('Contact2'))
FKRope = Function.deserialize(kindyn.fk('rope_anchor2'))

# Inverse Dynamics
ID = Function.deserialize(kindyn.rnea())

# OPTIMIZATION PARAMETERS
ns = 75  # number of shooting nodes

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

q, Q = create_variable('Q', nq, ns, 'STATE', 'SX')

foot_z_offset = 0.5

q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                  -0.3, -0.1, -0.1+foot_z_offset,  # Contact 1
                  -0.3, -0.05, -0.1+foot_z_offset,  # Contact 2
                  -1.57, -1.57, -3.1415,  # rope_anchor
                  0.3]).tolist()  # rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0,  # Floating base
                  0.3, 0.05, 0.01 +foot_z_offset,  # Contact 1
                  0.3, 0.1, 0.01 + foot_z_offset,  # Contact 2
                  1.57, 1.57, 3.1415,  # rope_anchor
                  4.0]).tolist()  # rope
alpha = 0.3# 0.3
rope_lenght = 0.3
#x_foot = rope_lenght * np.sin(alpha)
x_foot = 0.15
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   x_foot, 0., 0.+foot_z_offset,
                   x_foot, 0., 0.+foot_z_offset,
                   0., alpha, 0.,
                   rope_lenght]).tolist()
print "q_init: ", q_init

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
F_integrator = RK4_time(dae, 'SX')


# START WITH AN EMPTY NLP
X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, FRope, Dt])
V = concat_states_and_controls({"X": X, "U": U})
v_min, v_max = create_bounds({"x_min": [q_min, qdot_min], "x_max": [q_max, qdot_max],
                              "u_min": [qddot_min, f_min1, f_min2, f_minRope, dt_min], "u_max": [qddot_max, f_max1, f_max2, f_maxRope, dt_max]}, ns)

# SET UP COST FUNCTION
J = SX([0])

min_qdot = lambda k: .01*dot(Qdot[k], Qdot[k])
J += cost_function(min_qdot, 0, ns)

min_qddot = lambda k: 10.*dot(Qddot[k], Qddot[k])
#J += cost_function(min_qddot, 0, ns-1)

min_F = lambda k: 10.*dot(F1[k]+F2[k], F1[k]+F2[k])
#J += cost_function(min_F, 0, ns-1)

# K = 1000.
# min_qd = lambda k:  K*dot(Q[k][0:7]-q_init[0:7], Q[k][0:7]-q_init[0:7])# + K*dot(Q[k][3:7]-q_init[3:7], Q[k][3:7]-q_init[3:7])
# J += cost_function(min_qd, 0, ns)

# rope_init_lenght = 0.3
#
# jump_length = 2.0
#
# q_final = np.array([0.0, 0.0, -jump_length, 0.0, 0.0, 0.0, 1.0,
#                    x_foot, 0., 0.+foot_z_offset,
#                    x_foot, 0., 0.+foot_z_offset,
#                    0., alpha, 0.,
#                    rope_init_lenght+jump_length]).tolist()



q_trg = np.array([-.3, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0,
                  2.0]).tolist()

K = 1.
#min_qd = lambda k: K*dot(Q[k][0]-q_trg[0], Q[k][0]-q_trg[0]) + K*dot(Q[k][3:7]-q_trg[3:7], Q[k][3:7]-q_trg[3:7]) + K*dot(Q[k][-1]-q_trg[-1], Q[k][-1]-q_trg[-1])
min_qd = lambda k: K*dot(Q[k][0]-q_trg[0], Q[k][0]-q_trg[0]) + K*dot(Q[k][3:7]-q_trg[3:7], Q[k][3:7]-q_trg[3:7]) + K*dot(Q[k][-1]-q_trg[-1], Q[k][-1]-q_trg[-1])
J += cost_function(min_qd, 0, ns)


min_dt = lambda k: 1.*dot(Dt[k],Dt[k])
#J += cost_function(min_dt, 0, ns-1)

# SET UP CONSTRAINTS
G = constraint_handler()

# INITIAL CONDITION CONSTRAINT
x_init = cons.initial_condition.initial_condition(X[0], q_init + qdot_init)
g1, g_min1, g_max1 = constraint(x_init, 0, 1)
G.set_constraint(g1, g_min1, g_max1)


# MULTIPLE SHOOTING CONSTRAINT
integrator_dict = {'x0': X, 'p': Qddot, 'time': Dt}
multiple_shooting_constraint = multiple_shooting(integrator_dict, F_integrator)

g2, g_min2, g_max2 = constraint(multiple_shooting_constraint, 0, ns-1)
G.set_constraint(g2, g_min2, g_max2)

# INVERSE DYNAMICS CONSTRAINT
# dd = {'rope_anchor2': FRope}
dd = {'rope_anchor2': FRope, 'Contact1': F1, 'Contact2': F2}
id = inverse_dynamics(Q, Qdot, Qddot, ID, dd, kindyn, kindyn.LOCAL_WORLD_ALIGNED)

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
mu = 1.0
R_wall = np.zeros([3, 3])
R_wall[0, 2] = 1.0
R_wall[1, 1] = 1.0
R_wall[2, 0] = -1.0

surface_dict = {'a': 1., 'd': -x_foot}
Jac1 = Function.deserialize(kindyn.jacobian('Contact1', kindyn.LOCAL_WORLD_ALIGNED))
Jac2 = Function.deserialize(kindyn.jacobian('Contact2', kindyn.LOCAL_WORLD_ALIGNED))
JacRope = Function.deserialize(kindyn.jacobian('rope_anchor2', kindyn.LOCAL_WORLD_ALIGNED))


#FIRST 10 NODES THE ROBOT IS IN CONTACT
initial_stance_nodes = 5

contact_handler_F1 = cons.contact.contact_handler(FKR, F1)
#contact_handler_F1.setContact(Q, q_init)
#contact_handler_F1.setContactAndFrictionCone(Q, q_init, mu, R_wall)
contact_handler_F1.setSurfaceContactAndFrictionCone(Q, surface_dict, Jac1, Qdot, mu, R_wall)
g4, g_min4, g_max4 = constraint(contact_handler_F1, 0, ns)
# G.set_constraint(g4, g_min4, g_max4)

contact_handler_F2 = cons.contact.contact_handler(FKL, F2)
#contact_handler_F2.setContact(Q, q_init)
#contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_wall)
contact_handler_F2.setSurfaceContactAndFrictionCone(Q, surface_dict, Jac2, Qdot, mu, R_wall)
g5, g_min5, g_max5 = constraint(contact_handler_F2, 0, ns)
# G.set_constraint(g5, g_min5, g_max5)

contact_handler_FRope = cons.contact.contact_handler(FKRope, FRope)
#contact_handler_F2.setContact(Q, q_init)
#contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_wall)
contact_handler_FRope.setSurfaceContact(surface_dict, Q, JacRope, Qdot)
g5, g_min5, g_max5 = constraint(contact_handler_FRope, 0, ns)
#G.set_constraint(g5, g_min5, g_max5)


#ACTIONS
stance_F1 = cons.contact.contact_handler(FKR, F1)
stance_F1.setSurfaceContactAndFrictionCone(Q, surface_dict, Jac1, Qdot, mu, R_wall)
stance_F2 = cons.contact.contact_handler(FKL, F2)
stance_F2.setSurfaceContactAndFrictionCone(Q, surface_dict, Jac2, Qdot, mu, R_wall)

fly_F1 = cons.contact.contact_handler(FKR, F1)
fly_F1.removeContact()
fly_F2 = cons.contact.contact_handler(FKL, F2)
fly_F2.removeContact()

actions_dict = collections.OrderedDict()
actions_dict['R'] = [stance_F1, fly_F2] #R stance
actions_dict['D1'] = [stance_F1, stance_F2] #double stance
actions_dict['L'] = [fly_F1, stance_F2] #L stance
actions_dict['D2'] = [stance_F1, stance_F2] #double stance


start_walking_node = initial_stance_nodes
action_phases = 3  # [['R' 'D1' 'L' 'D2'] ...]
nodes_per_action = 5


footsep_scheduler = footsteps_scheduler(start_walking_node, action_phases, nodes_per_action, ns, actions_dict)
footsep_scheduler.printInfo()
g, gmin, gmax = footsep_scheduler.get_constraints()
G.set_constraint([g], gmin, gmax)

# AFTER
g, gmin, gmax = constraint(stance_F1, footsep_scheduler.getEndingNode()+1, ns)
G.set_constraint(g, gmin, gmax)
g, gmin, gmax = constraint(stance_F2, footsep_scheduler.getEndingNode()+1, ns)
G.set_constraint(g, gmin, gmax)
######

# BEFORE
g, gmin, gmax = constraint(stance_F1, 0, initial_stance_nodes)
G.set_constraint(g, gmin, gmax)
g, gmin, gmax = constraint(stance_F2, 0, initial_stance_nodes)
G.set_constraint(g, gmin, gmax)
################


g, gmin, gmax = G.get_constraints()

opts = {#'ipopt.tol': 0.1,
        #'ipopt.constr_viol_tol': 0.1,
        'ipopt.hessian_approximation': 'exact',
        'ipopt.max_iter': 4000,
        #'ipopt.dual_inf_tol': 10,
        'ipopt.linear_system_scaling': 'mc19',
        'ipopt.nlp_scaling_method': 'gradient-based',
        'ipopt.linear_solver': 'ma57'}

g_, g_min_, g_max_ = G.get_constraints()
solver = nlpsol('solver', 'ipopt', {'f': J, 'x': V, 'g': g_}, opts)

x0 = create_init({"x_init": [q_init, qdot_init], "u_init": [qddot_init, f_init1, f_init2, f_initRope, dt_init]}, ns)

sol = solver(x0=x0, lbx=v_min, ubx=v_max, lbg=g_min_, ubg=g_max_)
w_opt_tmp = sol['x'].full().flatten()
lam_g = sol['lam_g'].full().flatten()
lam_x = sol['lam_x'].full().flatten()
print (sol)

solver2 = nlpsol('solver2', 'ipopt', {'f': J, 'x': V, 'g': g_}, opts)
sol2 = solver2(x0=w_opt_tmp, lbx=v_min, ubx=v_max, lbg=g_min_, ubg=g_max_, lam_x0=lam_x, lam_g0=lam_g)
w_opt = sol2['x'].full().flatten()



# RETRIEVE SOLUTION AND LOGGING
solution_dict = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'FRope': FRope, 'Dt': Dt}, w_opt)

q_hist = solution_dict['Q']
q_hist = normalize_quaternion(q_hist)
F1_hist = solution_dict['F1']
F2_hist = solution_dict['F2']
Qddot_hist = solution_dict['Qddot']


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


BaseLink_pos = FKcomputer.computeFK('base_link', 'ee_pos', 0, ns)
get_BaseLink_pos = Function("base_link", [V], [BaseLink_pos], ['V'], ['BaseLink_pos'])
BaseLink_pos_hist = (get_BaseLink_pos(V=w_opt)['BaseLink_pos'].full().flatten()).reshape(ns, 3)

BaseLink_vel_angular = FKcomputer.computeDiffFK('base_link', 'ee_vel_angular', kindyn.LOCAL_WORLD_ALIGNED, 0, ns)
get_BaseLink_vel_angular = Function("get_BaseLink_vel_angular", [V], [BaseLink_vel_angular], ['V'], ['BaseLink_vel_angular'])
BaseLink_vel_angular_hist = (get_BaseLink_vel_angular(V=w_opt)['BaseLink_vel_angular'].full().flatten()).reshape(ns, 3)

BaseLink_vel_linear = FKcomputer.computeDiffFK('base_link', 'ee_vel_linear', kindyn.LOCAL_WORLD_ALIGNED, 0, ns)
get_BaseLink_vel_linear = Function("get_BaseLink_vel_linear", [V], [BaseLink_vel_linear], ['V'], ['BaseLink_vel_linear'])
BaseLink_vel_linear_hist = (get_BaseLink_vel_linear(V=w_opt)['BaseLink_vel_linear'].full().flatten()).reshape(ns, 3)

AnchorPoint_pos = FKcomputer.computeFK('rope_anchor2', 'ee_pos', 0, ns)
get_AnchorPoint_pos = Function('rope_anchor2', [V], [AnchorPoint_pos], ['V'], ['AnchorPoint_pos'])
AnchorPoint_pos_hist = (get_AnchorPoint_pos(V=w_opt)['AnchorPoint_pos'].full().flatten()).reshape(ns, 3)

AnchorPoint_vel = FKcomputer.computeDiffFK('rope_anchor2', 'ee_vel_linear', kindyn.LOCAL_WORLD_ALIGNED, 0, ns)
get_AnchorPoint_vel_lin = Function('get_rope_anchor2_vel_lin', [V], [AnchorPoint_vel], ['V'], ['AnchorPoint_vel'])
AnchorPoint_vel_hist = (get_AnchorPoint_vel_lin(V=w_opt)['AnchorPoint_vel'].full().flatten()).reshape(ns, 3)



# RESAMPLE STATE FOR REPLAY TRAJECTORY
dt = 0.001
#X_res, Tau_res = resample_integrator(X, Qddot, dt_hist, dt, dae, ID, dd, kindyn, kindyn.LOCAL_WORLD_ALIGNED)
X_res, U_res, Tau_res = resample_integrator_with_controls(X, U, Qddot, dt_hist, dt, dae, ID, dd, kindyn, kindyn.LOCAL_WORLD_ALIGNED)
get_X_res = Function("get_X_res", [V], [X_res], ['V'], ['X_res'])
x_hist_res = get_X_res(V=w_opt)['X_res'].full()
q_hist_res = (x_hist_res[0:nq, :]).transpose()
qdot_hist_res = (x_hist_res[nq:nq+nv, :]).transpose()

get_Tau_res = Function("get_Tau_res", [V], [Tau_res], ['V'], ['Tau_res'])
tau_hist_res = get_Tau_res(V=w_opt)['Tau_res'].full().transpose()



get_U_res = Function("get_U_res", [V], [U_res], ['V'], ['U_res'])
u_hist_res = get_U_res(V=w_opt)['U_res'].full()
F1_hist_res = (u_hist_res[nv:nv+3, :]).transpose()
F2_hist_res = (u_hist_res[nv+3:nv+6, :]).transpose()


logger.add('Q_res', q_hist_res)
logger.add('Qdot_res', qdot_hist_res)
logger.add('F1_res', F1_hist_res)
logger.add('F2_res', F2_hist_res)
logger.add('F1_hist', F1_hist)
logger.add('F2_hist', F2_hist)
logger.add('Tau_hist', tau_hist)
logger.add('Tau_res', tau_hist_res)
logger.add('Tf', tf)
logger.add('Contact1_pos_hist', Contact1_pos_hist)
logger.add('Contact2_pos_hist', Contact2_pos_hist)
logger.add('Waist_pos', Waist_pos_hist)
logger.add('Waist_rot', Waist_rot_hist)
logger.add('BaseLink_pos_hist', BaseLink_pos_hist)
logger.add('BaseLink_vel_ang_hist', BaseLink_vel_angular_hist)
logger.add('BaseLink_vel_lin_hist', BaseLink_vel_linear_hist)
logger.add('AnchorPoint_pos_hist', AnchorPoint_pos_hist)
logger.add('AnchorPoint_vel_hist', AnchorPoint_vel_hist)
logger.add('qddot_hist', Qddot_hist)


del(logger)

# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

#contact_dict = {'Contact1': F1_hist_res, 'Contact2': F2_hist_res}
#replay_trajectory(dt, joint_list, q_hist_res, contact_dict).replay()

replay_trajectory(dt, joint_list, q_hist_res).replay()