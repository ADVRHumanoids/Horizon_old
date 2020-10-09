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

Time = 0.5

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
f_min1 = (-1000.*np.ones(nf)).tolist()
f_max1 = (1000.*np.ones(nf)).tolist()
f_init1 = np.zeros(nf).tolist()

f2, F2 = create_variable('F2', nf, ns, 'CONTROL', 'SX')
f_min2 = (-1000.*np.ones(nf)).tolist()
f_min2 = np.array([-1000., -1000., 200.]).tolist()
f_max2 = (1000.*np.ones(nf)).tolist()
f_init2 = np.zeros(nf).tolist()

f3, F3 = create_variable('F3', nf, ns, 'CONTROL', 'SX')
f_min3 = (-1000.*np.ones(nf)).tolist()
f_max3 = (1000.*np.ones(nf)).tolist()
f_init3 = np.zeros(nf).tolist()

f4, F4 = create_variable('F4', nf, ns, 'CONTROL', 'SX')
f_min4 = (-1000.*np.ones(nf)).tolist()
f_min4 = np.array([-1000., -1000., 200.]).tolist()
f_max4 = (1000.*np.ones(nf)).tolist()
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
# x_init = q_init + qdot_init
# init = cons.initial_condition.initial_condition(X[0], x_init)
# g1, g_min1, g_max1 = constraint(init, 0, 1)
# G.set_constraint(g1, g_min1, g_max1)

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

torque_lims = cons.torque_limits.torque_lims(id, tau_min, tau_max)
g3, g_min3, g_max3 = constraint(torque_lims, 0, ns-1)
G.set_constraint(g3, g_min3, g_max3)


# WALL
mu = 0.01

R_ground = np.identity(3, dtype=float)

# STANCE ON 2 FEET
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

g, g_min, g_max = G.get_constraints()
x0 = create_init({"x_init": [q_init, qdot_init], "u_init": [qddot_init, f_init1, f_init2, f_init3, f_init4, dt_init]}, ns)

# IPOPT
opts_ipopt = \
    {
    'ipopt.tol': 0.001,
    'ipopt.constr_viol_tol': 0.001,
    'ipopt.max_iter': 5000,
    'ipopt.linear_solver': 'ma57'
    }

solver_ipopt = nlpsol('solver', 'ipopt', {'f': J, 'x': V, 'g': g}, opts_ipopt)
t_ipopt = time.time()
sol_ipopt = solver_ipopt(x0=x0, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
elapsed_ipopt = time.time() - t_ipopt
w_opt_ipopt = sol_ipopt['x'].full().flatten()

# SQP
opts_sqp = {'max_iter': 1}

J_sqp = V-w_opt_ipopt

solver_sqp = sqp('solver', "osqp", {'f': J_sqp, 'x': V, 'g': []}, opts_sqp)
t_sqp = time.time()
solution_sqp = solver_sqp(x0=w_opt_ipopt, lbx=v_min, ubx=v_max, lbg=[], ubg=[])
elapsed_sqp = time.time() - t_sqp
w_opt_sqp = solution_sqp['x']

# RETRIEVE SOLUTION AND LOGGING
solution_dict_ipopt = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'Dt': Dt}, w_opt_ipopt)

Q_ipopt = solution_dict_ipopt['Q']
Q_ipopt = normalize_quaternion(Q_ipopt)
F1_ipopt = solution_dict_ipopt['F1']
F2_ipopt = solution_dict_ipopt['F2']
F3_ipopt = solution_dict_ipopt['F3']
F4_ipopt = solution_dict_ipopt['F4']
Qdot_ipopt = solution_dict_ipopt['Qdot']
Qddot_ipopt = solution_dict_ipopt['Qddot']
Dt_ipopt = solution_dict_ipopt['Dt']


solution_dict_sqp = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'Dt': Dt}, w_opt_sqp)
Q_sqp = solution_dict_sqp['Q']
Q_sqp = normalize_quaternion(Q_sqp)
F1_sqp = solution_dict_sqp['F1']
F2_sqp = solution_dict_sqp['F2']
F3_sqp = solution_dict_sqp['F3']
F4_sqp = solution_dict_sqp['F4']
Qdot_sqp = solution_dict_sqp['Qdot']
Qddot_sqp = solution_dict_sqp['Qddot']
Dt_sqp = solution_dict_sqp['Dt']

Tau = id.compute_nodes(0, ns-1)
get_Tau = Function("get_Tau", [V], [Tau], ['V'], ['Tau'])
Tau_ipopt = (get_Tau(V=w_opt_ipopt)['Tau'].full().flatten()).reshape(ns-1, nv)
Tau_sqp = (get_Tau(V=w_opt_sqp)['Tau'].full().flatten()).reshape(ns-1, nv)

logger.add('Q_ipopt', Q_ipopt)
logger.add('Qdot_ipopt', Qdot_ipopt)
logger.add('Qddot_ipopt', Qddot_ipopt)
logger.add('F1_ipopt', F1_ipopt)
logger.add('F2_ipopt', F2_ipopt)
logger.add('F3_ipopt', F3_ipopt)
logger.add('F4_ipopt', F4_ipopt)
logger.add('Tau_ipopt', Tau_ipopt)
logger.add('Dt_ipopt', Dt_ipopt)
logger.add('elapsed_ipopt', elapsed_ipopt)

logger.add('Q_sqp', Q_sqp)
logger.add('Qdot_sqp', Qdot_sqp)
logger.add('Qddot_sqp', Qddot_sqp)
logger.add('F1_sqp', F1_sqp)
logger.add('F2_sqp', F2_sqp)
logger.add('F3_sqp', F3_sqp)
logger.add('F4_sqp', F4_sqp)
logger.add('Tau_sqp', Tau_sqp)
logger.add('Dt_sqp', Dt_sqp)
logger.add('elapsed_sqp', elapsed_sqp)

# MPC LOOP

mpc_iter = 100

sol_mpc = w_opt_ipopt

Q_mpc = np.zeros((mpc_iter, nq))
Qdot_mpc = np.zeros((mpc_iter, nv))
Qddot_mpc = np.zeros((mpc_iter, nv))
F1_mpc = np.zeros((mpc_iter, nf))
F2_mpc = np.zeros((mpc_iter, nf))
F3_mpc = np.zeros((mpc_iter, nf))
F4_mpc = np.zeros((mpc_iter, nf))
Tau_mpc = np.zeros((mpc_iter, nv))
elapsed_mpc = np.zeros((mpc_iter, 1))

G_mpc = constraint_handler()

g2, g_min2, g_max2 = constraint(multiple_shooting_constraint, 0, ns-1)
G_mpc.set_constraint(g2, g_min2, g_max2)

torque_lims = cons.torque_limits.torque_lims(id, tau_min, tau_max)
g3, g_min3, g_max3 = constraint(torque_lims, 0, ns-1)
G_mpc.set_constraint(g3, g_min3, g_max3)

# STANCE ON 2 FEET
contact_handler_F1 = cons.contact.contact_handler(FK1, F1)
contact_handler_F1.removeContact()
g8, g_min8, g_max8 = constraint(contact_handler_F1, 0, ns-1)
G_mpc.set_constraint(g8, g_min8, g_max8)

contact_handler_F2 = cons.contact.contact_handler(FK2, F2)
contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g5, g_min5, g_max5 = constraint(contact_handler_F2, 0, ns-1)
G_mpc.set_constraint(g5, g_min5, g_max5)

contact_handler_F3 = cons.contact.contact_handler(FK3, F3)
contact_handler_F3.removeContact()
g10, g_min10, g_max10 = constraint(contact_handler_F3, 0, ns-1)
G_mpc.set_constraint(g10, g_min10, g_max10)

contact_handler_F4 = cons.contact.contact_handler(FK4, F4)
contact_handler_F4.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g7, g_min7, g_max7 = constraint(contact_handler_F4, 0, ns-1)
G_mpc.set_constraint(g7, g_min7, g_max7)

g_mpc, g_min_mpc, g_max_mpc = G_mpc.get_constraints()

J_mpc = SX([0])

# J_mpc = V - w_opt_ipopt

min_q = lambda k: 10.0*dot(Q[k]-q_init, Q[k]-q_init)
J_mpc += cost_function(min_q,  0, ns)

min_qdot = lambda k: 10.*dot(Qdot[k], Qdot[k])
J_mpc += cost_function(min_qdot,  0, ns)

min_jerk = lambda k: 0.001*dot(Qddot[k]-Qddot[k-1], Qddot[k]-Qddot[k-1])
J_mpc += cost_function(min_jerk, 0, ns-1) # <- this smooths qddot solution

min_deltaFC = lambda k: 0.001*dot((F2[k]-F2[k-1])+(F4[k]-F4[k-1]),
                                 +(F2[k]-F2[k-1])+(F4[k]-F4[k-1]))  # min Fdot
J_mpc += cost_function(min_deltaFC, 0, ns-1)

# solver_ipopt_mpc = nlpsol('solver', "ipopt", {'f':  dot(J_mpc, J_mpc), 'x': V, 'g': g}, opts_ipopt)
# solver_ipopt_mpc = nlpsol('solver', "ipopt", {'f':  dot(J_mpc, J_mpc), 'x': V, 'g': g_mpc}, opts_ipopt)
solver_ipopt_mpc = nlpsol('solver', "ipopt", {'f':  J_mpc, 'x': V, 'g': g_mpc}, opts_ipopt)
solver_sqp_mpc = sqp('solver', "osqp", {'f': J_mpc, 'x': V, 'g': g_mpc}, opts_sqp)
# solver_sqp_mpc = sqp('solver', "osqp", {'f': J_mpc, 'x': V, 'g': g}, opts_sqp)

for k in range(mpc_iter):

    print 'mpc iter', k

    t_mpc = time.time()
    # solution_mpc = solver_ipopt_mpc(x0=sol_mpc, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
    solution_mpc = solver_ipopt_mpc(x0=sol_mpc, lbx=v_min, ubx=v_max, lbg=g_min_mpc, ubg=g_max_mpc)
    # solution_mpc = solver_sqp_mpc(x0=sol_mpc, lbx=v_min, ubx=v_max, lbg=g_min_mpc, ubg=g_max_mpc)
    # solution_mpc = solver_sqp_mpc(x0=sol_mpc, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
    elapsed_mpc[k] = time.time() - t_mpc

    sol_mpc = solution_mpc['x']

    # RETRIEVE SOLUTION AND LOGGING
    solution_dict_mpc = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4,
                                          'Dt': Dt}, sol_mpc)

    q_sol = solution_dict_mpc['Q']
    q_sol = normalize_quaternion(q_sol)
    qdot_sol = solution_dict_mpc['Qdot']
    qddot_sol = solution_dict_mpc['Qddot']
    F1_sol = solution_dict_mpc['F1']
    F2_sol = solution_dict_mpc['F2']
    F3_sol = solution_dict_mpc['F3']
    F4_sol = solution_dict_mpc['F4']
    tau_sol = (get_Tau(V=sol_mpc)['Tau'].full().flatten()).reshape(ns - 1, nv)

    Q_mpc[k, :] = q_sol[0, :]
    Qdot_mpc[k, :] = qdot_sol[0, :]
    Qddot_mpc[k, :] = qddot_sol[0, :]
    F1_mpc[k, :] = F1_sol[0, :]
    F2_mpc[k, :] = F2_sol[0, :]
    F3_mpc[k, :] = F3_sol[0, :]
    F4_mpc[k, :] = F4_sol[0, :]
    Tau_mpc[k, :] = tau_sol[0, :]

    # UPDATE
    v_min[0:nq] = q_min
    v_max[0:nq] = q_max
    v_min[nq:nq + nv] = v_max[nq:nq + nv] = qdot_sol[1, :]

    # DISTURBANCE ON FB TWIST
    dist = np.random.rand(6)-0.5*np.ones(6)

    v_min[nq:nq+6] = v_max[nq:nq+6] = qdot_sol[1, 0:6] + 0.1*dist

print 'END MPC'

logger.add('Q_mpc', Q_mpc)
logger.add('Qdot_mpc', Qdot_mpc)
logger.add('Qddot_mpc', Qddot_mpc)
logger.add('F1_mpc', F1_mpc)
logger.add('F2_mpc', F2_mpc)
logger.add('F3_mpc', F3_mpc)
logger.add('F4_mpc', F4_mpc)
logger.add('Tau_mpc', Tau_mpc)
logger.add('elapsed_mpc', elapsed_mpc)

del(logger)

# REPLAY TRAJECTORY MPC LOOP
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'Contact3_x', 'Contact3_y', 'Contact3_z',
              'Contact4_x', 'Contact4_y', 'Contact4_z']

contact_dict = {'Contact1': F1_mpc, 'Contact2': F2_mpc, 'Contact3': F3_mpc, 'Contact4': F4_mpc}
dt = 0.4
replay = replay_trajectory(dt, joint_list, Q_mpc, contact_dict, kindyn)
replay.sleep(2.)
replay.replay()
