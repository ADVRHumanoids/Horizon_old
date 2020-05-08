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

FREE_FALL = False

logger = []
if FREE_FALL:
    logger = matl.MatLogger2('/tmp/free_fall_log')
else:
    logger = matl.MatLogger2('/tmp/rope_suspended_log')
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
ns = 30  # number of shooting nodes

nc = 3  # number of contacts

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)

DoF = nq - 7  # Contacts + anchor_rope + rope

nv = kindyn.nv()  # Velocity DoFs

nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

# CREATE VARIABLES
q, Q = create_variable("Q", nq, ns, "STATE")

q_min = []
q_max = []
if FREE_FALL:
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
else:
    q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0,  # Floating base
                      -0.3, -0.1, -0.1,  # Contact 1
                      -0.3, -0.05, -0.1,  # Contact 2
                      -1.57, -1.57, -3.1415,  # rope_anchor
                      0.1]).tolist()  # rope
    q_max = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0,  # Floating base
                      0.3, 0.05, 0.1,  # Contact 1
                      0.3, 0.1, 0.1,  # Contact 2
                      1.57, 1.57, 3.1415,  # rope_anchor
                      0.1]).tolist()  # rope
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

tf = 1.0  # [s]

# FORMULATE DISCRETE TIME DYNAMICS
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
opts = {'tf': tf/ns}
F_integrator = RK4(dae, opts, "SX")

# START WITH AN EMPTY NLP
X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, FRope])
V = concat_states_and_controls({"X": X, "U": U})
v_min, v_max = create_bounds({"x_min": [q_min, qdot_min], "x_max": [q_max, qdot_max],
                              "u_min": [qddot_min, f_min1, f_min2, f_minRope], "u_max": [qddot_max, f_max1, f_max2, f_maxRope]}, ns)

# SET UP COST FUNCTION
J = SX([0])
#
min_qdot = lambda k: 100.*dot(Qdot[k][6:-1], Qdot[k][6:-1])
J += cost_function(min_qdot, 0, ns)
#
min_qddot_a = lambda k: 1000.*dot(Qddot[k][6:-1], Qddot[k][6:-1])
J += cost_function(min_qddot_a, 0, ns-1)
#
min_F1 = lambda k: 1000.*dot(F1[k], F1[k])
J += cost_function(min_F1, 0, ns-1)
#
min_F2 = lambda k: 1000.*dot(F2[k], F2[k])
J += cost_function(min_F2, 0, ns-1)
#
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
integrator_dict = {'x0': X, 'p': Qddot}
multiple_shooting_constraint = multiple_shooting(integrator_dict, F_integrator)
g2, g_min2, g_max2 = constraint(multiple_shooting_constraint, 0, ns-1)
G.set_constraint(g2, g_min2, g_max2)

# INVERSE DYNAMICS CONSTRAINT
dd = {'rope_anchor2': FRope}
id = inverse_dynamics(Q, Qdot, Qddot, ID, dd, kindyn)

tau_min = []
tau_max = []
if FREE_FALL:
    tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                    -1000., -1000., -1000.,  # Contact 1
                    -1000., -1000., -1000.,  # Contact 2
                    0., 0., 0.,  # rope_anchor
                    0.]).tolist()  # rope
else:
    tau_min = np.array([0., 0., 0., 0., 0., 0.,  # Floating base
                        -1000., -1000., -1000.,  # Contact 1
                        -1000., -1000., -1000.,  # Contact 2
                        0., 0., 0.,  # rope_anchor
                        -10000.0]).tolist()  # rope

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

x0 = create_init({"x_init": [q_init, qdot_init], "u_init": [qddot_init, f_init1, f_init2, f_initRope]}, ns)


sol = solver(x0=x0, lbx=v_min, ubx=v_max, lbg=g_min, ubg=g_max)
w_opt = sol['x'].full().flatten()


# RETRIEVE SOLUTION AND LOGGING
solution_dict = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'FRope': FRope}, w_opt)
q_hist = solution_dict['Q']


# RESAMPLE STATE FOR REPLAY TRAJECTORY
dt = 0.001
X_res, Tau_res = resample_integrator(X, Qddot, tf, dt, dae, ID, dd, kindyn)
get_X_res = Function("get_X_res", [V], [X_res], ['V'], ['X_res'])
x_hist_res = get_X_res(V=w_opt)['X_res'].full()
q_hist_res = (x_hist_res[0:nq, :]).transpose()

get_Tau_res = Function("get_Tau_res", [V], [Tau_res], ['V'], ['Tau_res'])
tau_hist_res = get_Tau_res(V=w_opt)['Tau_res'].full().transpose()

# GET ADDITIONAL VARIABLES
Tau = id.compute_nodes(0, ns-1)
get_Tau = Function("get_Tau", [V], [Tau], ['V'], ['Tau'])
tau_hist = (get_Tau(V=w_opt)['Tau'].full().flatten()).reshape(ns-1, nv)

# LOGGING
for k in solution_dict:
    logger.add(k, solution_dict[k])

#logger.add('Q_res', q_hist_res)
#logger.add('Tau_res', tau_hist_res)

FKcomputer = kinematics(kindyn, Q, Qdot, Qddot)
ContactRope_pos = FKcomputer.computeFK('rope_anchor2', 'ee_pos', 0, ns)
get_ContactRope_pos = Function("get_ContactRope_pos", [V], [ContactRope_pos], ['V'], ['ContactRope_pos'])
ContactRope_pos_hist = (get_ContactRope_pos(V=w_opt)['ContactRope_pos'].full().flatten()).reshape(ns, 3)


logger.add('ContactRope_pos_hist', ContactRope_pos_hist)
logger.add('q_hist', q_hist)
logger.add('tau_hist', tau_hist)

time = np.arange(0.0, tf, tf/ns)
logger.add('time', time)

del(logger)

#### PLOTS ####
PLOT = True
if PLOT:


    if FREE_FALL:
        plt.figure(1) ### Rope anchor point
        plt.plot(time, ContactRope_pos_hist[:,0], label='$\mathrm{x}$', linewidth=3.0)
        plt.plot(time, ContactRope_pos_hist[:,1], label='$\mathrm{y}$', linewidth=3.0)
        plt.plot(time, ContactRope_pos_hist[:,2], label='$\mathrm{z}$', linewidth=3.0)
        plt.legend(loc='upper right', fancybox=True, framealpha=0.5, prop={'size':20})
        plt.grid()
        plt.suptitle('$\mathrm{Anchor-Point \ Position}$', size = 20)
        plt.xlabel('$\mathrm{[sec]}$', size = 20)
        plt.ylabel('$\mathrm{[m]}$', size = 20)
        plt.savefig("free_fall_anchor_point_pos.pdf", format="pdf")

        plt.figure(2) ### Rope lenght Vs floating base position z

        plt.plot(time, q_hist[:,2], label='$\mathrm{floating-base \ z}$', linewidth=3.0, color='red')
        plt.plot(time[29], q_hist[29,2], marker='o', markersize=7, color='red')
        plt.plot(time[0], q_hist[0,2],  marker='o', markersize=7, color='red')
        plt.text(time[29], q_hist[29,2]-0.2,'('+str(round(time[29],2))+','+str(round(q_hist[29,2],2))+')', horizontalalignment='right', verticalalignment='top')
        plt.text(time[0], q_hist[0,2]-0.2,'('+str(round(time[0],2))+','+str(round(q_hist[0,2],2))+')', horizontalalignment='left', verticalalignment='top')

        plt.plot(time, q_hist[:,-1], label='$\mathrm{rope \ joint}$', linewidth=3.0, color='blue')
        plt.plot(time[29], q_hist[29,-1], marker='o', markersize=7, color='blue')
        plt.plot(time[0], q_hist[0,-1],  marker='o', markersize=7, color='blue')
        plt.text(time[29], q_hist[29,-1]+0.2,'('+str(round(time[29],2))+','+str(round(q_hist[29,-1],2))+')', horizontalalignment='right', verticalalignment='bottom')
        plt.text(time[0], q_hist[0,-1]+0.2,'('+str(round(time[0],2))+','+str(round(q_hist[0,-1],2))+')', horizontalalignment='left', verticalalignment='bottom')

        plt.legend(loc='upper left', fancybox=True, framealpha=0.5, prop={'size':20})
        plt.grid()
        plt.suptitle('$\mathrm{Rope \ Lenght \ vs \ Floating-Base \ Z \ Position}$', size = 20)
        plt.xlabel('$\mathrm{[sec]}$', size = 20)
        plt.ylabel('$\mathrm{[m]}$', size = 20)

        plt.savefig("free_fall_rope_lenght_vs_floating_base_z_pos.pdf", format="pdf")
    else:
        plt.figure(1,figsize=(10, 8))  ### Rope anchor point
        plt.suptitle('$\mathrm{Rope \ Joint}$', size=20)
        s1 = plt.subplot(211)
        s1.set_title('$\mathrm{Position}$', size=20)
        axes = plt.gca()
        axes.set_ylim([0.0, 0.2])
        axes.set_yticks([0.0, 0.1, 0.2])
        axes.set_aspect(1.5)
        plt.plot(time, q_hist[:, -1], linewidth=3.0)
        plt.grid()
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[m]}$', size=20)

        s2 = plt.subplot(212)
        s2.set_title('$\mathrm{Force}$', size=20)
        axes = plt.gca()
        axes.set_ylim([-598.5, -598.3])
        axes.set_yticks([-598.5, -598.4, -598.3])
        axes.set_aspect(1.5)
        axes.ticklabel_format(useOffset=False)
        plt.plot(time[0:29], np.around(tau_hist[:, -1],decimals = 1), linewidth=3.0)
        plt.grid()
        plt.xlabel('$\mathrm{[sec]}$', size=20)
        plt.ylabel('$\mathrm{[N]}$', size=20)
        plt.savefig("rope_suspended_rope_joint.pdf", format="pdf")

    plt.show()
###



# REPLAY TRAJECTORY
joint_list = ['Contact1_x', 'Contact1_y', 'Contact1_z',
              'Contact2_x', 'Contact2_y', 'Contact2_z',
              'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
              'rope_joint']

replay_trajectory(dt, joint_list, q_hist_res).replay()