#!/usr/bin/env python

import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import horizon
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn
import matlogger2.matlogger as matl
from Horizon.constraints import initial_condition
from Horizon.constraints import torque_limits
from Horizon.constraints import contact
from Horizon.constraints import position
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

print("q_init: ", q_init)

qdot, Qdot = create_variable('Qdot', nv, ns, 'STATE', 'SX')
qdot_min = (-100.*np.ones(nv)).tolist()
qdot_max = (100.*np.ones(nv)).tolist()
qdot_init = np.zeros(nv).tolist()

qddot, Qddot = create_variable('Qddot', nv, ns, 'CONTROL', 'SX')
qddot_min = (-100.*np.ones(nv)).tolist()
qddot_max = (100.*np.ones(nv)).tolist()
qddot_init = np.zeros(nv).tolist()
qddot_init[2] = -9.8

f2, F2 = create_variable('F2', nf, ns, 'CONTROL', 'SX')
f_min2 = (-1000.*np.ones(nf)).tolist()
f_min2 = np.array([-1000., -1000., 200.]).tolist()
f_max2 = (1000.*np.ones(nf)).tolist()
f_init2 = np.zeros(nf).tolist()

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
X, U = create_state_and_control([Q, Qdot], [Qddot, F2, F4, Dt])
V = concat_states_and_controls({"X": X, "U": U})
v_min, v_max = create_bounds({"x_min": [q_min, qdot_min], "x_max": [q_max, qdot_max],
                              "u_min": [qddot_min, f_min2, f_min4, dt_min], "u_max": [qddot_max, f_max2, f_max4, dt_max]}, ns)

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
v_min[nq:nq+nv] = v_max[nq:nq+nv] = qdot_init

# MULTIPLE SHOOTING CONSTRAINT
integrator_dict = {'x0': X, 'p': Qddot, 'time': Dt}
multiple_shooting_constraint = multiple_shooting(integrator_dict, F_integrator)

g1, g_min1, g_max1 = constraint(multiple_shooting_constraint, 0, ns-1)
G.set_constraint(g1, g_min1, g_max1)

# INVERSE DYNAMICS CONSTRAINT
dd = {'Contact2': F2, 'Contact4': F4}
id = inverse_dynamics(Q, Qdot, Qddot, ID, dd, kindyn, kindyn.LOCAL_WORLD_ALIGNED)

tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    -1000., -1000., -1000.,  # Contact 1
                    -1000., -1000., -1000.,  # Contact 2
                    -1000., -1000., -1000.,  # Contact 3
                    -1000., -1000., -1000.]).tolist()  # Contact 4

tau_max = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    1000., 1000., 1000.,  # Contact 1
                    1000., 1000., 1000.,  # Contact 2
                    1000., 1000., 1000.,  # Contact 3
                    1000., 1000., 1000.]).tolist()  # Contact 4

torque_lims = torque_limits.torque_lims_fb(id)
g2, g_min2, g_max2 = constraint(torque_lims, 0, ns-1)
G.set_constraint(g2, g_min2, g_max2)

mu = 0.01
R_ground = np.identity(3, dtype=float)

# STANCE ON 2 FEET
ground_z = FK1(q=q_init)['ee_pos'][2]
C1_ws = position.position(FK1, Q, [-10., -10, ground_z+0.2], [10., 10, 10.])
g3, g_min3, g_max3 = constraint(C1_ws, 0, ns-1)
G.set_constraint(g3, g_min3, g_max3)

C3_ws = position.position(FK3, Q, [-10., -10, ground_z+0.2], [10., 10, 10.])
g4, g_min4, g_max4 = constraint(C3_ws, 0, ns-1)
G.set_constraint(g4, g_min4, g_max4)

contact_handler_F2 = contact.contact_handler(FK2, F2)
contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g5, g_min5, g_max5 = constraint(contact_handler_F2, 0, ns-1)
G.set_constraint(g5, g_min5, g_max5)

contact_handler_F4 = contact.contact_handler(FK4, F4)
contact_handler_F4.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g6, g_min6, g_max6 = constraint(contact_handler_F4, 0, ns-1)
G.set_constraint(g6, g_min6, g_max6)

g, g_min, g_max = G.get_constraints()
x0 = create_init({"x_init": [q_init, qdot_init], "u_init": [qddot_init, f_init2, f_init4, dt_init]}, ns)

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

# RETRIEVE SOLUTION AND LOGGING
solution_dict_ipopt = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F2': F2, 'F4': F4, 'Dt': Dt}, w_opt_ipopt)

Q_ipopt = solution_dict_ipopt['Q']
Q_ipopt = normalize_quaternion(Q_ipopt)
F2_ipopt = solution_dict_ipopt['F2']
F4_ipopt = solution_dict_ipopt['F4']
Qdot_ipopt = solution_dict_ipopt['Qdot']
Qddot_ipopt = solution_dict_ipopt['Qddot']
Dt_ipopt = solution_dict_ipopt['Dt']

Tau = id.compute_nodes(0, ns-1)
get_Tau = Function("get_Tau", [V], [Tau], ['V'], ['Tau'])
Tau_ipopt = (get_Tau(V=w_opt_ipopt)['Tau'].full().flatten()).reshape(ns-1, nv)

logger.add('Q_ipopt', Q_ipopt)
logger.add('Qdot_ipopt', Qdot_ipopt)
logger.add('Qddot_ipopt', Qddot_ipopt)
logger.add('F2_ipopt', F2_ipopt)
logger.add('F4_ipopt', F4_ipopt)
logger.add('Tau_ipopt', Tau_ipopt)
logger.add('Dt_ipopt', Dt_ipopt)
logger.add('elapsed_ipopt', elapsed_ipopt)

FKcomputer = kinematics(kindyn, Q, Qdot, Qddot)
Contact1_pos = FKcomputer.computeFK('Contact1', 'ee_pos', 0, ns)
get_Contact1_pos = Function("get_Contact1_pos", [V], [Contact1_pos], ['V'], ['Contact1_pos'])

Contact2_pos = FKcomputer.computeFK('Contact2', 'ee_pos', 0, ns)
get_Contact2_pos = Function("get_Contact2_pos", [V], [Contact2_pos], ['V'], ['Contact2_pos'])

Contact3_pos = FKcomputer.computeFK('Contact3', 'ee_pos', 0, ns)
get_Contact3_pos = Function("get_Contact3_pos", [V], [Contact3_pos], ['V'], ['Contact3_pos'])

Contact4_pos = FKcomputer.computeFK('Contact4', 'ee_pos', 0, ns)
get_Contact4_pos = Function("get_Contact4_pos", [V], [Contact4_pos], ['V'], ['Contact4_pos'])

com_pos = FKcomputer.computeCoM('com', 0, ns)
get_com_pos = Function("get_com_pos", [V], [com_pos], ['V'], ['com_pos'])

Waist_pos = FKcomputer.computeFK('Waist', 'ee_pos', 0, ns)
get_Waist_pos = Function("get_Waist_pos", [V], [Waist_pos], ['V'], ['Waist_pos'])

Waist_rot = FKcomputer.computeFK('Waist', 'ee_rot', 0, ns)
get_Waist_rot = Function("get_Waist_rot", [V], [Waist_rot], ['V'], ['Waist_rot'])

# MPC LOOP

mpc_iter = 150

Q_mpc = np.zeros((mpc_iter, nq))
Qdot_mpc = np.zeros((mpc_iter, nv))
Qddot_mpc = np.zeros((mpc_iter, nv))
F1_mpc = np.zeros((mpc_iter, nf))
F2_mpc = np.zeros((mpc_iter, nf))
F3_mpc = np.zeros((mpc_iter, nf))
F4_mpc = np.zeros((mpc_iter, nf))
C1_mpc = np.zeros((mpc_iter, 3))
C2_mpc = np.zeros((mpc_iter, 3))
C3_mpc = np.zeros((mpc_iter, 3))
C4_mpc = np.zeros((mpc_iter, 3))
Tau_mpc = np.zeros((mpc_iter, nv))
CoM_mpc = np.zeros((mpc_iter, 3))
Waist_pos_mpc = np.zeros((mpc_iter, 3))
Waist_rot_mpc = np.zeros((mpc_iter, 3))
elapsed_mpc = np.zeros((mpc_iter, 1))

# IPOPT
G_mpc_ipopt = constraint_handler()

g1, g_min1, g_max1 = constraint(multiple_shooting_constraint, 0, ns-1)
G_mpc_ipopt.set_constraint(g1, g_min1, g_max1)

g2, g_min2, g_max2 = constraint(torque_lims, 0, ns-1)
G_mpc_ipopt.set_constraint(g2, g_min2, g_max2)

# STANCE ON 2 FEET
g3, g_min3, g_max3 = constraint(C1_ws, 0, ns-1)
G_mpc_ipopt.set_constraint(g3, g_min3, g_max3)

g4, g_min4, g_max4 = constraint(C3_ws, 0, ns-1)
G_mpc_ipopt.set_constraint(g4, g_min4, g_max4)

contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g5, g_min5, g_max5 = constraint(contact_handler_F2, 0, ns-1)
G_mpc_ipopt.set_constraint(g5, g_min5, g_max5)

contact_handler_F4.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g6, g_min6, g_max6 = constraint(contact_handler_F4, 0, ns-1)
G_mpc_ipopt.set_constraint(g6, g_min6, g_max6)

g_mpc_ipopt, g_min_mpc_ipopt, g_max_mpc_ipopt = G_mpc_ipopt.get_constraints()

J_mpc_ipopt = SX([0])

min_q = lambda k: 100.0*dot(Q[k]-Q_ipopt[0], Q[k]-Q_ipopt[0])
J_mpc_ipopt += cost_function(min_q,  0, ns)

min_qdot = lambda k: 100.*dot(Qdot[k][0:6], Qdot[k][0:6])
J_mpc_ipopt += cost_function(min_qdot,  0, ns)

min_qddot = lambda k: 0.001*dot(Qddot[k][7:-1], Qddot[k][7:-1])
# J_mpc_ipopt += cost_function(min_qddot,  0, ns-1)

min_jerk = lambda k: 0.0001*dot(Qddot[k]-Qddot[k-1], Qddot[k]-Qddot[k-1])
J_mpc_ipopt += cost_function(min_jerk, 0, ns-1) # <- this smooths qddot solution

min_deltaFC = lambda k: 10.*dot((F2[k]-F2[k-1])+(F4[k]-F4[k-1]),
                                (F2[k]-F2[k-1])+(F4[k]-F4[k-1]))  # min Fdot
J_mpc_ipopt += cost_function(min_deltaFC, 0, ns-1)

solver_ipopt_mpc = nlpsol('solver', "ipopt", {'f':  J_mpc_ipopt, 'x': V, 'g': g_mpc_ipopt}, opts_ipopt)

# SQP
G_mpc_sqp = constraint_handler()

g1, g_min1, g_max1 = constraint(multiple_shooting_constraint, 0, ns-1)
G_mpc_sqp.set_constraint(g1, g_min1, g_max1)

g2, g_min2, g_max2 = constraint(torque_lims, 0, ns-1)
G_mpc_sqp.set_constraint(g2, g_min2, g_max2)

# STANCE ON 2 FEET
g3, g_min3, g_max3 = constraint(C1_ws, 0, ns-1)
# G_mpc_sqp.set_constraint(g3, g_min3, g_max3)

g4, g_min4, g_max4 = constraint(C3_ws, 0, ns-1)
# G_mpc_sqp.set_constraint(g4, g_min4, g_max4)

contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g5, g_min5, g_max5 = constraint(contact_handler_F2, 0, ns-1)
# G_mpc_sqp.set_constraint(g5, g_min5, g_max5)

contact_handler_F4.setContactAndFrictionCone(Q, q_init, mu, R_ground)
g6, g_min6, g_max6 = constraint(contact_handler_F4, 0, ns-1)
# G_mpc_sqp.set_constraint(g6, g_min6, g_max6)

g_mpc_sqp, g_min_mpc_sqp, g_max_mpc_sqp = G_mpc_sqp.get_constraints()

J_mpc_sqp = []

J1 = 1e5*(vertcat(*g1)-vertcat(*g_min1))
J_mpc_sqp = vertcat(J_mpc_sqp, J1)

J2 = 1e5*(vertcat(*g2)-vertcat(*g_min2))
# J_mpc_sqp = vertcat(J_mpc_sqp, J2)

C1_init = FK2(q=q_init)['ee_pos']
C1_ws = position.position(FK1, Q, [C1_init[0], C1_init[1], C1_init[2]], [C1_init[0], C1_init[1],  C1_init[2]])
g3, g_min3, g_max3 = constraint(C1_ws, 0, ns)
J3 = 1e1*(vertcat(*g3)-vertcat(*g_min3))
J_mpc_sqp = vertcat(J_mpc_sqp, J3)

C2_init = FK2(q=q_init)['ee_pos']
C2_ws = position.position(FK2, Q, C2_init, C2_init)
g4, g_min4, g_max4 = constraint(C2_ws, 0, ns)
J4 = 1e6*(vertcat(*g4)-vertcat(*g_min4))
J_mpc_sqp = vertcat(J_mpc_sqp, J4)

C3_init = FK3(q=q_init)['ee_pos']
C3_ws = position.position(FK3, Q, [C3_init[0], C3_init[1], C3_init[2]], [C3_init[0], C3_init[1],  C3_init[2]])
g5, g_min5, g_max5 = constraint(C3_ws, 0, ns)
J5 = 1e1*(vertcat(*g5)-vertcat(*g_min5))
J_mpc_sqp = vertcat(J_mpc_sqp, J5)

C4_init = FK4(q=q_init)['ee_pos']
C4_ws = position.position(FK4, Q, C4_init, C4_init)
g6, g_min6, g_max6 = constraint(C4_ws, 0, ns)
J6 = 1e6*(vertcat(*g6)-vertcat(*g_min6))
J_mpc_sqp = vertcat(J_mpc_sqp, J6)

J7 = 100.*vertcat(*Qdot)
# J_mpc_sqp = vertcat(J_mpc_sqp, J7)

J8 = 100.*vertcat(*Qddot)
# J_mpc_sqp = vertcat(J_mpc_sqp, J8)

J9 = 1000.*(vertcat(*Q) - vertcat(*Q_ipopt))
J_mpc_sqp = vertcat(J_mpc_sqp, J9)

J10 = 1000.*(vertcat(*Qdot) - vertcat(*Qdot_ipopt))
J_mpc_sqp = vertcat(J_mpc_sqp, J10)

J11 = 1.*((vertcat(*F2) - vertcat(*F2_ipopt)) + (vertcat(*F4) - vertcat(*F4_ipopt)))
J_mpc_sqp = vertcat(J_mpc_sqp, J11)

opts_sqp = {'max_iter': 1}

solver_sqp_mpc = sqp('solver', "osqp", {'f': J_mpc_sqp, 'x': V, 'g': g_mpc_sqp}, opts_sqp)

sol_mpc = w_opt_ipopt

print('START MPC LOOP')

for k in range(mpc_iter):

    print('MPC ITER', k)

    t_mpc = time.time()
    # IPOPT
    # solution_mpc = solver_ipopt_mpc(x0=sol_mpc, lbx=v_min, ubx=v_max, lbg=g_min_mpc_ipopt, ubg=g_max_mpc_ipopt)
    # SQP
    solution_mpc = solver_sqp_mpc(x0=sol_mpc, lbx=v_min, ubx=v_max, lbg=g_min_mpc_sqp, ubg=g_max_mpc_sqp)
    elapsed_mpc[k] = time.time() - t_mpc

    sol_mpc = solution_mpc['x']

    # RETRIEVE SOLUTION AND LOGGING
    solution_dict_mpc = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F2': F2, 'F4': F4,
                                          'Dt': Dt}, sol_mpc)

    q_sol = solution_dict_mpc['Q']
    q_sol = normalize_quaternion(q_sol)
    qdot_sol = solution_dict_mpc['Qdot']
    qddot_sol = solution_dict_mpc['Qddot']
    F2_sol = solution_dict_mpc['F2']
    F4_sol = solution_dict_mpc['F4']
    tau_sol = (get_Tau(V=sol_mpc)['Tau'].full().flatten()).reshape(ns - 1, nv)

    C1_sol = (get_Contact1_pos(V=sol_mpc)['Contact1_pos'].full().flatten()).reshape(ns, 3)
    C2_sol = (get_Contact2_pos(V=sol_mpc)['Contact2_pos'].full().flatten()).reshape(ns, 3)
    C3_sol = (get_Contact3_pos(V=sol_mpc)['Contact3_pos'].full().flatten()).reshape(ns, 3)
    C4_sol = (get_Contact4_pos(V=sol_mpc)['Contact4_pos'].full().flatten()).reshape(ns, 3)
    CoM_sol = (get_com_pos(V=sol_mpc)['com_pos'].full().flatten()).reshape(ns, 3)

    Waist_pos_sol = (get_Waist_pos(V=sol_mpc)['Waist_pos'].full().flatten()).reshape(ns, 3)
    Waist_rot_sol = (get_Waist_rot(V=sol_mpc)['Waist_rot'].full().flatten()).reshape(ns, 3, 3)
    Waist_rot_sol = rotation_matrix_to_euler(Waist_rot_sol)

    Q_mpc[k, :] = q_sol[1, :]
    Qdot_mpc[k, :] = qdot_sol[1, :]
    Qddot_mpc[k, :] = qddot_sol[1, :]
    F2_mpc[k, :] = F2_sol[1, :]
    F4_mpc[k, :] = F4_sol[1, :]
    C1_mpc[k, :] = C1_sol[1, :]
    C2_mpc[k, :] = C2_sol[1, :]
    C3_mpc[k, :] = C3_sol[1, :]
    C4_mpc[k, :] = C4_sol[1, :]
    Waist_pos_mpc[k, :] = Waist_pos_sol[1, :]
    Waist_rot_mpc[k, :] = Waist_rot_sol[1, :]
    Tau_mpc[k, :] = tau_sol[1, :]
    CoM_mpc[k, :] = CoM_sol[1, :]

    # UPDATE
    v_min[nq:nq + nv] = v_max[nq:nq + nv] = qdot_sol[1, :]

    # DISTURBANCE ON FB TWIST
    # dist = np.random.rand(6)-0.5*np.ones(6)
    dist = 0.1*np.array([1, 1, 1, 1, 1, 1])
    dist_select = np.diag([0, 1, 0, 0, 0, 0])

    # if k % 20 == 0 and k > 0:
    if 20 <= k <= 25:
        v_min[nq:nq+6] = v_max[nq:nq+6] = qdot_sol[1, 0:6] + mtimes(dist_select, dist)
    if 40 <= k <= 45:
        v_min[nq:nq + 6] = v_max[nq:nq + 6] = qdot_sol[1, 0:6] - mtimes(dist_select, dist)

print('END MPC LOOP')

logger.add('Q_mpc', Q_mpc)
logger.add('Qdot_mpc', Qdot_mpc)
logger.add('Qddot_mpc', Qddot_mpc)
logger.add('F1_mpc', F1_mpc)
logger.add('F2_mpc', F2_mpc)
logger.add('F3_mpc', F3_mpc)
logger.add('F4_mpc', F4_mpc)
logger.add('C1_mpc', C1_mpc)
logger.add('C2_mpc', C2_mpc)
logger.add('C3_mpc', C3_mpc)
logger.add('C4_mpc', C4_mpc)
logger.add('Tau_mpc', Tau_mpc)
logger.add('CoM_mpc', CoM_mpc)
logger.add('Waist_pos_mpc', Waist_pos_mpc)
logger.add('Waist_rot_mpc', Waist_rot_mpc)
logger.add('elapsed_mpc', elapsed_mpc)

del(logger)

# REPLAY TRAJECTORY MPC LOOP
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'Contact3_x', 'Contact3_y', 'Contact3_z',
              'Contact4_x', 'Contact4_y', 'Contact4_z']

contact_dict = {'Contact1': F1_mpc, 'Contact2': F2_mpc, 'Contact3': F3_mpc, 'Contact4': F4_mpc}
dt = dt_init
replay = replay_trajectory(dt, joint_list, Q_mpc, contact_dict, kindyn)
replay.sleep(1.)
replay.replay()
