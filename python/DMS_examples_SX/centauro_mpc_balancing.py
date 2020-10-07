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
from utils.conversions_to_euler import *
from utils.dt_RKF import *

from solvers.sqp import *

logger = matl.MatLogger2('/tmp/centauro_mpc_balancing_log')
logger.setBufferMode(matl.BufferMode.CircularBuffer)

urdf = rospy.get_param('robot_description')
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

# Forward Kinematics of interested links
FK_waist = Function.deserialize(kindyn.fk('Waist'))
FK1 = Function.deserialize(kindyn.fk('Contact1'))
FK2 = Function.deserialize(kindyn.fk('Contact2'))
FK3 = Function.deserialize(kindyn.fk('Contact3'))
FK4 = Function.deserialize(kindyn.fk('Contact4'))

# Inverse Dynamics
ID = Function.deserialize(kindyn.rnea())

# OPTIMIZATION PARAMETERS
ns = 10  # number of shooting nodes

nc = 4  # number of contacts

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)

DoF = nq - 7  # Contacts + anchor_rope + rope

nv = kindyn.nv()  # Velocity DoFs

nf = 3  # Force DOfs

Time = 0.1

# CREATE VARIABLES
dt, Dt = create_variable('Dt', 1, ns, 'CONTROL', 'SX')
dt_min = Time/ns
dt_max = Time/ns
dt_init = dt_min

q, Q = create_variable('Q', nq, ns, 'STATE', 'SX')

# CENTAURO homing

disp_z = 0.2

q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                  0.1, 0.1, -0.635,
                  0.1, -0.5, -0.635,
                  -0.6, -0.5, -0.635,
                  -0.6, 0.1, -0.635]).tolist()

q_max = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0,  # Floating base
                  0.6, 0.5, -0.635 + disp_z,
                  0.6, -0.1, -0.635 + disp_z,
                  -0.1, -0.1, -0.635 + disp_z,
                  -0.1, 0.5, -0.635 + disp_z]).tolist()

q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0.349999, 0.349999, -0.635,
                   0.349999, -0.349999, -0.635,
                   -0.349999, -0.349999, -0.635,
                   -0.349999, 0.349999, -0.635]).tolist()

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

f3, F3 = create_variable('F3', nf, ns, 'CONTROL', 'SX')
f_min3 = (-10000.*np.ones(nf)).tolist()
f_max3 = (10000.*np.ones(nf)).tolist()
f_init3 = np.zeros(nf).tolist()

f4, F4 = create_variable('F4', nf, ns, 'CONTROL', 'SX')
f_min4 = (-10000.*np.ones(nf)).tolist()
f_max4 = (10000.*np.ones(nf)).tolist()
f_init4 = np.zeros(nf).tolist()

x, xdot = dynamic_model_with_floating_base(q, qdot, qddot)

# FORMULATE DISCRETE TIME DYNAMICS
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': []}
F_integrator = RK4_time(dae, 'SX')

# START WITH AN EMPTY NLP
X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, F3, F4, Dt])
V = concat_states_and_controls({"X": X, "U": U})
v_min, v_max = create_bounds({"x_min": [q_min, qdot_min], "x_max": [q_max, qdot_max],
                              "u_min": [qddot_min, f_min1, f_min2, f_min3, f_min4, dt_min], "u_max": [qddot_max, f_max1, f_max2, f_max3, f_max4, dt_max]}, ns)

# SET UP COST FUNCTION
J = SX([0])

min_q = lambda k: 1.0*dot(Q[k]-q_init, Q[k]-q_init)
J += cost_function(min_q,  0, ns)

min_qdot = lambda k: 10.*dot(Qdot[k], Qdot[k])
J += cost_function(min_qdot,  0, ns)

min_jerk = lambda k: 0.001*dot(Qddot[k]-Qddot[k-1], Qddot[k]-Qddot[k-1])
J += cost_function(min_jerk, 0, ns-1) # <- this smooths qddot solution

min_deltaFC = lambda k: 0.001*dot((F2[k]-F2[k-1])+(F4[k]-F4[k-1]),
                                 +(F2[k]-F2[k-1])+(F4[k]-F4[k-1]))  # min Fdot
J += cost_function(min_deltaFC, 0, ns-1)

# CONSTRAINTS
G = constraint_handler()

# INITIAL CONDITION CONSTRAINT
v_min[0:nq] = v_max[0:nq] = q_init
v_min[nq:nq+nv] = v_max[nq:nq+nv] = qdot_init

# MULTIPLE SHOOTING CONSTRAINT
integrator_dict = {'x0': X, 'p': Qddot, 'time': Dt}
multiple_shooting_constraint = multiple_shooting(integrator_dict, F_integrator)

g2, g_min2, g_max2 = constraint(multiple_shooting_constraint, 0, ns-1)
G.set_constraint(g2, g_min2, g_max2)

# INVERSE DYNAMICS CONSTRAINT
dd = {'Contact1': F1, 'Contact2': F2, 'Contact3': F3, 'Contact4': F4}
id = inverse_dynamics(Q, Qdot, Qddot, ID, dd, kindyn, kindyn.LOCAL_WORLD_ALIGNED)

tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    -10000., -10000., -10000.,  # Contact 1
                    -10000., -10000., -10000.,  # Contact 2
                    -10000., -10000., -10000.,  # Contact 3
                    -10000., -10000., -10000.]).tolist()  # Contact 4

tau_max = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    10000., 10000., 10000.,  # Contact 1
                    10000., 10000., 10000.,  # Contact 2
                    10000., 10000., 10000.,  # Contact 3
                    10000., 10000., 10000.]).tolist()  # Contact 4

torque_lims1 = cons.torque_limits.torque_lims(id, tau_min, tau_max)
g3, g_min3, g_max3 = constraint(torque_lims1, 0, ns-1)
G.set_constraint(g3, g_min3, g_max3)


# WALL
mu = 0.01

R_ground = np.identity(3, dtype=float)

# STANCE PHASE
contact_handler_F1 = cons.contact.contact_handler(FK1, F1)
contact_handler_F1.removeContact()
g8, g_min8, g_max8 = constraint(contact_handler_F1, 0, ns-1)
G.set_constraint(g8, g_min8, g_max8)

contact_handler_F2 = cons.contact.contact_handler(FK2, F2)
contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g5, g_min5, g_max5 = constraint(contact_handler_F2, 0, ns-1)
G.set_constraint(g5, g_min5, g_max5)

contact_handler_F3 = cons.contact.contact_handler(FK3, F3)
contact_handler_F3.removeContact()
g10, g_min10, g_max10 = constraint(contact_handler_F3, 0, ns-1)
G.set_constraint(g10, g_min10, g_max10)

contact_handler_F4 = cons.contact.contact_handler(FK4, F4)
contact_handler_F4.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g7, g_min7, g_max7 = constraint(contact_handler_F4, 0, ns-1)
G.set_constraint(g7, g_min7, g_max7)

g_, g_min_, g_max_ = G.get_constraints()
x0 = create_init({"x_init": [q_init, qdot_init], "u_init": [qddot_init, f_init1, f_init2, f_init3, f_init4, dt_init]}, ns)

opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 5000,
        'ipopt.linear_solver': 'ma57'}

solver = nlpsol('solver', 'ipopt', {'f': J, 'x': V, 'g': g_}, opts)
t_ipopt = time.time()
sol = solver(x0=x0, lbx=v_min, ubx=v_max, lbg=g_min_, ubg=g_max_)
elapsed_ipopt = time.time() - t_ipopt
w_opt_ipopt = sol['x'].full().flatten()

# SQP
opts = {'max_iter': 1}

J_sqp = V-w_opt_ipopt

solver = sqp('solver', "osqp", {'f': J_sqp, 'x': V, 'g': []}, opts)
t_sqp = time.time()
solution = solver(x0=w_opt_ipopt, lbx=v_min, ubx=v_max, lbg=[], ubg=[])
elapsed_sqp = time.time() - t_sqp

print "elapsed_ipopt: ", elapsed_ipopt
print "elapsed_sqp: ", elapsed_sqp

obj_history = solution['f']
print "obj_history: ", obj_history
con_history = solution['g']
w_opt = solution['x']

# RETRIEVE SOLUTION AND LOGGING
solution_dict = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'Dt': Dt}, w_opt)

q_hist = solution_dict['Q']
q_hist = normalize_quaternion(q_hist)
F1_hist = solution_dict['F1']
F2_hist = solution_dict['F2']
F3_hist = solution_dict['F3']
F4_hist = solution_dict['F4']
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

Contact3_pos = FKcomputer.computeFK('Contact3', 'ee_pos', 0, ns)
get_Contact3_pos = Function("get_Contact3_pos", [V], [Contact3_pos], ['V'], ['Contact3_pos'])
Contact3_pos_hist = (get_Contact3_pos(V=w_opt)['Contact3_pos'].full().flatten()).reshape(ns, 3)

Contact4_pos = FKcomputer.computeFK('Contact4', 'ee_pos', 0, ns)
get_Contact4_pos = Function("get_Contact4_pos", [V], [Contact4_pos], ['V'], ['Contact4_pos'])
Contact4_pos_hist = (get_Contact4_pos(V=w_opt)['Contact4_pos'].full().flatten()).reshape(ns, 3)


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

# RESAMPLE STATE FOR REPLAY TRAJECTORY
dt = 0.001
#X_res, Tau_res = resample_integrator(X, Qddot, dt_hist, dt, dae, ID, dd, kindyn, kindyn.LOCAL_WORLD_ALIGNED)
X_res, U_res, Tau_res = resample_integrator_with_controls(X, U, Qddot, dt_hist, dt, dae, ID, dd, kindyn, kindyn.LOCAL_WORLD_ALIGNED)
get_X_res = Function("get_X_res", [V], [X_res], ['V'], ['X_res'])
x_hist_res = get_X_res(V=w_opt)['X_res'].full()
q_hist_res = (x_hist_res[0:nq, :]).transpose()

get_Tau_res = Function("get_Tau_res", [V], [Tau_res], ['V'], ['Tau_res'])
tau_hist_res = get_Tau_res(V=w_opt)['Tau_res'].full().transpose()

get_U_res = Function("get_U_res", [V], [U_res], ['V'], ['U_res'])
u_hist_res = get_U_res(V=w_opt)['U_res'].full()
F1_hist_res = (u_hist_res[nv:nv+3, :]).transpose()
F2_hist_res = (u_hist_res[nv+3:nv+6, :]).transpose()
F3_hist_res = (u_hist_res[nv+6:nv+9, :]).transpose()
F4_hist_res = (u_hist_res[nv+9:nv+12, :]).transpose()


logger.add('Q_res', q_hist_res)
logger.add('F1_res', F1_hist_res)
logger.add('F2_res', F2_hist_res)
logger.add('F3_res', F3_hist_res)
logger.add('F4_res', F4_hist_res)
logger.add('F1_hist', F1_hist)
logger.add('F2_hist', F2_hist)
logger.add('F3_hist', F3_hist)
logger.add('F4_hist', F4_hist)
logger.add('Tau_hist', tau_hist)
logger.add('Tau_res', tau_hist_res)
logger.add('Tf', tf)
logger.add('Contact1_pos_hist', Contact1_pos_hist)
logger.add('Contact2_pos_hist', Contact2_pos_hist)
logger.add('Contact3_pos_hist', Contact3_pos_hist)
logger.add('Contact4_pos_hist', Contact4_pos_hist)
logger.add('Waist_pos', Waist_pos_hist)
logger.add('Waist_rot', Waist_rot_hist)
logger.add('BaseLink_pos_hist', BaseLink_pos_hist)
logger.add('BaseLink_vel_ang_hist', BaseLink_vel_angular_hist)
logger.add('BaseLink_vel_lin_hist', BaseLink_vel_linear_hist)
logger.add('qddot_hist', Qddot_hist)

mpc_iter = 100

sol_mpc = w_opt

q_mpc = np.zeros((mpc_iter, nq))
F1_mpc = np.zeros((mpc_iter, nf))
F2_mpc = np.zeros((mpc_iter, nf))
F3_mpc = np.zeros((mpc_iter, nf))
F4_mpc = np.zeros((mpc_iter, nf))
Qddot_mpc = np.zeros((mpc_iter, nv))


integrator_dict = {'x0': X, 'p': Qddot, 'time': Dt}
multiple_shooting_constraint = multiple_shooting(integrator_dict, F_integrator)

G_mpc = constraint_handler()
g2, g_min2, g_max2 = constraint(multiple_shooting_constraint, 0, ns-1)
G_mpc.set_constraint(g2, g_min2, g_max2)

g_mpc, g_min_mpc, g_max_mpc = G_mpc.get_constraints()

opts_ipot_mpc = {'ipopt.tol': 0.001,
             'ipopt.constr_viol_tol': 0.001,
             'ipopt.max_iter': 10,
             'ipopt.linear_solver': 'ma57'}

# solver_mpc = nlpsol('solver', "ipopt", {'f': J, 'x': V, 'g': g_}, opts_ipot_mpc)

opts_sqp_mpc = {'max_iter': 1}
solver_mpc = sqp('solver', "osqp", {'f': J_sqp, 'x': V, 'g': g_mpc}, opts_sqp_mpc)

for k in range(mpc_iter):
    print 'mpc iter', k

    # dist = 0.1
    #
    # if k == 50:
    #     sol_mpc[0] += dist
    #     sol_mpc[7] -= dist
    #     sol_mpc[10] -= dist
    #     sol_mpc[13] -= dist
    #     sol_mpc[16] -= dist

    v_min[0:nq] = v_max[0:nq] = sol_mpc[0:nq]
    # v_min[nq:nq + nv] = v_max[nq:nq + nv] = sol_mpc[nq:(nq+nv)]

    solution_mpc = solver_mpc(x0=sol_mpc, lbx=v_min, ubx=v_max, lbg=g_min_, ubg=g_max_)
    # solution_mpc = solver_mpc(x0=sol_mpc, lbx=v_min, ubx=v_max, lbg=g_min_mpc, ubg=g_max_mpc)
    sol_mpc = solution_mpc['x']

    # RETRIEVE SOLUTION AND LOGGING
    solution_dict_mpc = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4,
                                          'Dt': Dt}, sol_mpc)

    q_sol = solution_dict_mpc['Q']
    q_sol = normalize_quaternion(q_sol)
    F1_sol = solution_dict_mpc['F1']
    F2_sol = solution_dict_mpc['F2']
    F3_sol = solution_dict_mpc['F3']
    F4_sol = solution_dict_mpc['F4']
    Qddot_sol = solution_dict_mpc['Qddot']

    q_mpc[k, :] = q_sol[1, :]
    F1_mpc[k, :] = F1_sol[1, :]
    F2_mpc[k, :] = F2_sol[1, :]
    F3_mpc[k, :] = F3_sol[1, :]
    F4_mpc[k, :] = F4_sol[1, :]
    Qddot_mpc[k, :] = Qddot_sol[1, :]

print 'END MPC'

logger.add('q_mpc', q_mpc)
logger.add('F1_mpc', F1_mpc)
logger.add('F2_mpc', F2_mpc)
logger.add('F3_mpc', F3_mpc)
logger.add('F4_mpc', F4_mpc)
logger.add('Qddot_mpc', Qddot_mpc)
logger.add('F2_sol', F2_sol)

del(logger)

# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'Contact3_x', 'Contact3_y', 'Contact3_z',
              'Contact4_x', 'Contact4_y', 'Contact4_z']

contact_dict = {'Contact1': F1_mpc, 'Contact2': F2_mpc, 'Contact3': F3_mpc, 'Contact4': F4_mpc}
dt = 0.01
replay = replay_trajectory(dt, joint_list, q_mpc, contact_dict, kindyn)
replay.sleep(2.)
replay.replay()
