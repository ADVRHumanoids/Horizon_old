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
import matplotlib.pyplot as plt


logger = matl.MatLogger2('/tmp/rope_jump_dt_log')
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
dt_max = 0.08 #0.08
dt_init = dt_min

q, Q = create_variable('Q', nq, ns, 'STATE', 'SX')

foot_z_offset = 0.#0.5

q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                  -0.3, -0.1, -0.1+foot_z_offset,  # Contact 1
                  -0.3, -0.05, -0.1+foot_z_offset,  # Contact 2
                  -1.57, -1.57, -3.1415,  # rope_anchor
                  0.3]).tolist()  # rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0,  # Floating base
                  0.3, 0.05, 0.1+foot_z_offset,  # Contact 1
                  0.3, 0.1, 0.1+foot_z_offset,  # Contact 2
                  1.57, 1.57, 3.1415,  # rope_anchor
                  0.3]).tolist()  # rope
alpha = 0.3# 0.3
rope_lenght = 0.3
x_foot = rope_lenght * np.sin(alpha)
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

opts = {'tol': 1.e-5}
#F_integrator = RKF_time(dae, opts, 'SX')
F_integrator2 = RKF_time(dae, opts, 'SX')

# START WITH AN EMPTY NLP
X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, FRope, Dt])
V = concat_states_and_controls({"X": X, "U": U})
v_min, v_max = create_bounds({"x_min": [q_min, qdot_min], "x_max": [q_max, qdot_max],
                              "u_min": [qddot_min, f_min1, f_min2, f_minRope, dt_min], "u_max": [qddot_max, f_max1, f_max2, f_maxRope, dt_max]}, ns)

lift_node = 2 #20
touch_down_node = 60

# SET UP COST FUNCTION
J = SX([0])

dict = {'x0':X, 'p':Qddot, 'time':Dt}
variable_time = dt_RKF(dict, F_integrator2)
Dt_RKF = variable_time.compute_nodes(0, ns-1)
# min_dt = lambda k: (Dt[k]-Dt_RKF[k])*(Dt[k]-Dt_RKF[k])
# J += cost_function(min_dt, 0, ns-1)

q_trg = np.array([-.4, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                  0.0, 0.0, 0.0+foot_z_offset,
                  0.0, 0.0, 0.0+foot_z_offset,
                  0.0, 0.0, 0.0,
                  0.3]).tolist()

K = 50.#1000.#6.5*1e5
min_qd = lambda k: K*dot(Q[k][0]-q_trg[0], Q[k][0]-q_trg[0])
J += cost_function(min_qd, lift_node+1, touch_down_node)

#x_init = q_init + qdot_init
#min_xinit = lambda k: 10.*dot(Qdot[k]-qdot_init, Qdot[k]-qdot_init)
#J += cost_function(min_xinit, touch_down_node+1, ns)


#min_qd2 = lambda k: 10.*dot(Q[k][3:7]-q_trg[3:7], Q[k][3:7]-q_trg[3:7])
#J += cost_function(min_qd2, lift_node+1, touch_down_node)
#
min_qdot = lambda k: 1.*dot(Qdot[k][6:12], Qdot[k][6:12])
J += cost_function(min_qdot, lift_node+1, ns)
#
#min_qddot = lambda k: .001*dot(Qddot[k], Qddot[k])
#J += cost_function(min_qddot, 0, ns-1)
#

min_jerk = lambda k: 0.001*dot(Qddot[k]-Qddot[k-1], Qddot[k]-Qddot[k-1])
J += cost_function(min_jerk, 0, ns-1) # <- this smooths qddot solution

# min_q = lambda k: 0.1*K*dot((Q[k]-q_init), (Q[k]-q_init))
# J += cost_function(min_q, touch_down_node, ns)

#min_FC = lambda k: 1.*dot(F1[k]+F2[k], F1[k]+F2[k])
#J += cost_function(min_FC, 0, lift_node)

min_deltaFC = lambda k: 1.*dot((F1[k]-F1[k-1])+(F2[k]-F2[k-1]), (F1[k]-F1[k-1])+(F2[k]-F2[k-1])) # min Fdot
J += cost_function(min_deltaFC, touch_down_node+1, ns-1)

#
#min_deltaFRope = lambda k: 0.02*dot(FRope[k]-FRope[k-1], FRope[k]-FRope[k-1])  # min Fdot
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

# # Test
# FK_bl = Function.deserialize(kindyn.fk('base_link'))
# pmin = [-1., -1., -1.]
# pmax = [0.05, 1., 1.]
# base_link_bb = cons.position.position(FK_bl, Q, pmin, pmax)
# g8, g_min8, g_max8 = constraint(base_link_bb, 0, ns)
# G.set_constraint(g8, g_min8, g_max8)


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
g9, g_min9, g_max9 = constraint(contact_handler_F1, 0, lift_node+1)
G.set_constraint(g9, g_min9, g_max9)

contact_handler_F2 = cons.contact.contact_handler(FKL, F2)
contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_wall)
g11, g_min11, g_max11 = constraint(contact_handler_F2, 0, lift_node+1)
G.set_constraint(g11, g_min11, g_max11)

# FLIGHT PHASE
contact_handler_F1.removeContact()
g12, g_min12, g_max12 = constraint(contact_handler_F1, lift_node+1, touch_down_node)
G.set_constraint(g12, g_min12, g_max12)

contact_handler_F2.removeContact()
g13, g_min13, g_max13 = constraint(contact_handler_F2, lift_node+1, touch_down_node)
G.set_constraint(g13, g_min13, g_max13)

# TOUCH DOWN
contact_handler_F1.setContactAndFrictionCone(Q, q_init, mu, R_wall)
g14, g_min14, g_max14 = constraint(contact_handler_F1, touch_down_node, ns)
G.set_constraint(g14, g_min14, g_max14)

contact_handler_F2.setContactAndFrictionCone(Q, q_init, mu, R_wall)
g1111, g_min1111, g_max1111 = constraint(contact_handler_F2, touch_down_node, ns)
G.set_constraint(g1111, g_min1111, g_max1111)

# FINAL CONDITION CONSTRAINT
#finalX = cons.initial_condition.state_condition(Q, q_init)
#g999, g_min999, g_max999 = constraint(finalX, touch_down_node+1, ns)
#G.set_constraint(g999, g_min999, g_max999)



opts = {'ipopt.tol': 0.001,
        'ipopt.constr_viol_tol': 0.001,
        'ipopt.max_iter': 5000,
        'ipopt.linear_solver': 'ma57'}

g_, g_min_, g_max_ = G.get_constraints()
solver = nlpsol('solver', 'ipopt', {'f': J, 'x': V, 'g': g_}, opts)

x0 = create_init({"x_init": [q_init, qdot_init], "u_init": [qddot_init, f_init1, f_init2, f_initRope, dt_init]}, ns)

sol = solver(x0=x0, lbx=v_min, ubx=v_max, lbg=g_min_, ubg=g_max_)
w_opt = sol['x'].full().flatten()

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


logger.add('Q_res', q_hist_res)
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

get_Dt_RKF = Function("get_Dt_RKF", [V], [Dt_RKF], ['V'], ['Dt_RKF'])
Dt_RKF_hist = get_Dt_RKF(V=w_opt)['Dt_RKF'].full().transpose()

logger.add('Dt_RKF', Dt_RKF_hist)
logger.add('qddot_hist', Qddot_hist)

goal = q_trg[0]*np.ones(ns)
logger.add('goal', goal)

del(logger)

#PLOTS
time  = [0]
labels = [str(time[0])]
ticks = [3,5,7,57,69, 70]
k = 1
for i in dt_hist:
    time.append(time[k-1] + i)
    if k in ticks:
        l = '%s' % float('%.2g' % time[k])
        labels.append(l)
    else:
        labels.append("")
    k+=1


plt.figure(1)
plt.suptitle('$\mathrm{Floating \ - \ Base \ X \ Trajectory}$', size=20)
plt.plot(time, BaseLink_pos_hist[:,0], linewidth=3.0, color='blue')
plt.plot(time, goal, linewidth=3.0, color='red', linestyle='--')
plt.xticks(time, labels)
axes = plt.gca()
axes.set_ylim([-0.5, 0.3])
plt.grid()
plt.xlabel('$\mathrm{[sec]}$', size=20)
plt.ylabel('$\mathrm{[m]}$', size=20)

plt.savefig("rope_jump_x_trj.pdf", format="pdf")

plt.figure(2)
plt.suptitle('$\mathrm{Floating \ Base \ v_x \ and \ \omega_y \ Trajectory}$', size=20)
plt.plot(time, BaseLink_vel_angular_hist[:,1], linewidth=3.0, color='blue', label='$\mathrm{\omega_{fb,y}}$' )
plt.plot(time, BaseLink_vel_linear_hist[:,0], linewidth=3.0, color='red', label='$\mathrm{\dot{p}_{fb,x}}$')
plt.xticks(time, labels)
plt.grid()
plt.xlabel('$\mathrm{[sec]}$', size=20)
plt.ylabel('$\mathrm{[\\frac{rad}{sec}]}$', size=20)
plt.legend(loc='lower right', fancybox=True, framealpha=0.5, prop={'size':20}, ncol=3)

plt.savefig("rope_jump_omega_x.pdf", format="pdf")

plt.figure(3, figsize=(15, 13))
plt.suptitle('$\mathrm{Feet: \ Contact \ Force \ X \ VS \ Position \ X}$', size=20)
ax1 = plt.subplot(211)
ax1.set_title('$\mathrm{Right}$', size=20)
plt.xticks(time, labels)
y1_color = 'black'
ax1.tick_params(axis='y', labelcolor=y1_color)
k = 0
for F1 in F1_hist:
    ax1.hlines(y=F1[0], xmin=time[k], xmax=time[k+1], linewidth=3, color=y1_color)
    k += 1
for k in range(1, len(time)-1):
    ax1.vlines(x=time[k], ymin=F1_hist[k-1][0], ymax=F1_hist[k][0], linewidth=3, color=y1_color)
ax1.grid()
ax1.set_xlabel('$\mathrm{[sec]}$', size=20)
ax1.set_ylabel('$\mathrm{[N]}$', size=20)


ax2 = ax1.twinx()
y2_color = 'blue'
ax2.tick_params(axis='y', labelcolor=y2_color)
ax2.plot(time, Contact1_pos_hist[:,0], linewidth=3.0, color=y2_color, linestyle='--')
ax2.set_ylabel('$\mathrm{[m]}$', size=20, color=y2_color)
##
ax1 = plt.subplot(212)
ax1.set_title('$\mathrm{Left}$', size=20)
y1_color = 'black'
ax1.tick_params(axis='y', labelcolor=y1_color)
plt.xticks(time, labels)
k = 0
for F2 in F2_hist:
    ax1.hlines(y=F2[0], xmin=time[k], xmax=time[k+1], linewidth=3, color=y1_color)
    k += 1
for k in range(1, len(time)-1):
    ax1.vlines(x=time[k], ymin=F2_hist[k-1][0], ymax=F2_hist[k][0], linewidth=3, color=y1_color)
ax1.grid()
ax1.set_xlabel('$\mathrm{[sec]}$', size=20)
ax1.set_ylabel('$\mathrm{[N]}$', size=20)

ax2 = ax1.twinx()
y2_color = 'blue'
ax2.tick_params(axis='y', labelcolor=y2_color)
ax2.plot(time, Contact2_pos_hist[:,0], linewidth=3.0, color=y2_color, linestyle='--')
ax2.set_ylabel('$\mathrm{[m]}$', size=20, color=y2_color)


plt.savefig("rope_jump_feet_norm_comparison.pdf", format="pdf")


plt.figure(4, figsize=(15, 13))
plt.suptitle('$\mathrm{Control \ Action}$', size=20)
###forces
ax1 = plt.subplot(211)
ax1.set_title('$\mathrm{Contact \ Forces}$', size=20)
plt.xticks(time, labels)
k = 0
for F1 in F1_hist:
    if k == 0:
        ax1.hlines(y=F1[0], xmin=time[k], xmax=time[k+1], linewidth=3, color='r', label='$\mathrm{F_{Cr,x}}$')
        ax1.hlines(y=F1[1], xmin=time[k], xmax=time[k + 1], linewidth=3, color='g', label='$\mathrm{F_{Cr,y}}$')
        ax1.hlines(y=F1[2], xmin=time[k], xmax=time[k + 1], linewidth=3, color='b', label='$\mathrm{F_{Cr,z}}$')
    else:
        ax1.hlines(y=F1[0], xmin=time[k], xmax=time[k + 1], linewidth=3, color='r')
        ax1.hlines(y=F1[1], xmin=time[k], xmax=time[k + 1], linewidth=3, color='g')
        ax1.hlines(y=F1[2], xmin=time[k], xmax=time[k + 1], linewidth=3, color='b')
    k += 1

k = 0
for F2 in F2_hist:
    if k == 0:
        ax1.hlines(y=F2[0], xmin=time[k], xmax=time[k+1], linewidth=3, color='c', linestyle='--', label='$\mathrm{F_{Cl,x}}$')
        ax1.hlines(y=F2[1], xmin=time[k], xmax=time[k + 1], linewidth=3, color='m', linestyle='--', label='$\mathrm{F_{Cl,y}}$')
        ax1.hlines(y=F2[2], xmin=time[k], xmax=time[k + 1], linewidth=3, color='k', linestyle='--', label='$\mathrm{F_{Cl,z}}$')
    else:
        ax1.hlines(y=F2[0], xmin=time[k], xmax=time[k + 1], linewidth=3, color='c', linestyle='--')
        ax1.hlines(y=F2[1], xmin=time[k], xmax=time[k + 1], linewidth=3, color='m', linestyle='--')
        ax1.hlines(y=F2[2], xmin=time[k], xmax=time[k + 1], linewidth=3, color='k', linestyle='--')
    k += 1

for k in range(1, len(time)-1):
    ax1.vlines(x=time[k], ymin=F1_hist[k-1][0], ymax=F1_hist[k][0], linewidth=3, color='r')
    ax1.vlines(x=time[k], ymin=F1_hist[k - 1][1], ymax=F1_hist[k][1], linewidth=3, color='g')
    ax1.vlines(x=time[k], ymin=F1_hist[k - 1][2], ymax=F1_hist[k][2], linewidth=3, color='b')

for k in range(1, len(time)-1):
    ax1.vlines(x=time[k], ymin=F2_hist[k-1][0], ymax=F2_hist[k][0], linewidth=3, color='c', linestyle='--')
    ax1.vlines(x=time[k], ymin=F2_hist[k - 1][1], ymax=F2_hist[k][1], linewidth=3, color='m', linestyle='--')
    ax1.vlines(x=time[k], ymin=F2_hist[k - 1][2], ymax=F2_hist[k][2], linewidth=3, color='k', linestyle='--')

ax1.grid()
ax1.set_xlabel('$\mathrm{[sec]}$', size=20)
ax1.set_ylabel('$\mathrm{[N]}$', size=20)
ax1.legend(loc='lower center', fancybox=True, framealpha=0.5, prop={'size':20}, ncol=3)

### acc
ax2 = plt.subplot(212)
ax2.set_title('$\mathrm{Leg \ Accelerations}$', size=20)
plt.xticks(time, labels)
k = 0
for qddot in Qddot_hist:
    if k == 0:
        ax2.hlines(y=qddot[6], xmin=time[k], xmax=time[k+1], linewidth=3, color='r', label='$\mathrm{\ddot{q}_{lr,1}}$')
        ax2.hlines(y=qddot[7], xmin=time[k], xmax=time[k + 1], linewidth=3, color='g', label='$\mathrm{\ddot{q}_{lr,2}}$')
        ax2.hlines(y=qddot[8], xmin=time[k], xmax=time[k + 1], linewidth=3, color='b', label='$\mathrm{\ddot{q}_{lr,3}}$')
    else:
        ax2.hlines(y=qddot[6], xmin=time[k], xmax=time[k + 1], linewidth=3, color='r')
        ax2.hlines(y=qddot[7], xmin=time[k], xmax=time[k + 1], linewidth=3, color='g')
        ax2.hlines(y=qddot[8], xmin=time[k], xmax=time[k + 1], linewidth=3, color='b')
    k += 1

k = 0
for qddot in Qddot_hist:
    if k == 0:
        ax2.hlines(y=qddot[9], xmin=time[k], xmax=time[k+1], linewidth=3, color='c', linestyle='--', label='$\mathrm{\ddot{q}_{ll,1}}$')
        ax2.hlines(y=qddot[10], xmin=time[k], xmax=time[k + 1], linewidth=3, color='m', linestyle='--', label='$\mathrm{\ddot{q}_{ll,2}}$')
        ax2.hlines(y=qddot[11], xmin=time[k], xmax=time[k + 1], linewidth=3, color='k', linestyle='--', label='$\mathrm{\ddot{q}_{ll,3}}$')
    else:
        ax2.hlines(y=qddot[9], xmin=time[k], xmax=time[k + 1], linewidth=3, color='c', linestyle='--')
        ax2.hlines(y=qddot[10], xmin=time[k], xmax=time[k + 1], linewidth=3, color='m', linestyle='--')
        ax2.hlines(y=qddot[11], xmin=time[k], xmax=time[k + 1], linewidth=3, color='k', linestyle='--')
    k += 1

for k in range(1, len(time)-1):
    ax2.vlines(x=time[k], ymin=Qddot_hist[k-1][6], ymax=Qddot_hist[k][6], linewidth=3, color='r')
    ax2.vlines(x=time[k], ymin=Qddot_hist[k - 1][7], ymax=Qddot_hist[k][7], linewidth=3, color='g')
    ax2.vlines(x=time[k], ymin=Qddot_hist[k - 1][8], ymax=Qddot_hist[k][8], linewidth=3, color='b')

for k in range(1, len(time)-1):
    ax2.vlines(x=time[k], ymin=Qddot_hist[k-1][9], ymax=Qddot_hist[k][9], linewidth=3, color='c', linestyle='--')
    ax2.vlines(x=time[k], ymin=Qddot_hist[k - 1][10], ymax=Qddot_hist[k][10], linewidth=3, color='m', linestyle='--')
    ax2.vlines(x=time[k], ymin=Qddot_hist[k - 1][11], ymax=Qddot_hist[k][11], linewidth=3, color='k', linestyle='--')

ax2.grid()
ax2.set_xlabel('$\mathrm{[sec]}$', size=20)
ax2.set_ylabel('$\mathrm{[\\frac{rad}{sec^2}]}$', size=20)
ax2.legend(loc='lower center', fancybox=True, framealpha=0.5, prop={'size':20}, ncol=3)



plt.savefig("rope_jump_force.pdf", format="pdf")




plt.show()





# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

contact_dict = {'Contact1': F1_hist_res, 'Contact2': F2_hist_res}
dt = 0.001
replay_trajectory(dt, joint_list, q_hist_res, contact_dict, kindyn).replay()
#replay_trajectory(dt, joint_list, q_hist_res).replay()
