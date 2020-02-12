#!/usr/bin/env python
from horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn
import matlogger2.matlogger as matl
import rospy
from constraints import *
from previewer import *
from inverse_dynamics import *
import math as mt

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

# Optimization Params
ns = 30  # number of shooting nodes

nc = 3  # number of contacts

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)

DoF = nq - 7 # Contacts + anchor_rope + rope

nv = kindyn.nv() # Velocity DoFs

nf = 3 # 2 feet contacts + rope contact with wall, Force DOfs

# Variables
q, Q = create_variable("Q", nq, ns, "STATE")

q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0, # Floating base
                  -0.3, -0.1, -0.1, # Contact 1
                  -0.3, -0.05, -0.1, # Contact 2
                  -1.57, -1.57, -3.1415, #rope_anchor
                  0.0]).tolist() #rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0, # Floating base
                  0.3, 0.05, 0.1,  # Contact 1
                  0.3, 0.1, 0.1,  # Contact 2
                  1.57, 1.57, 3.1415,  #rope_anchor
                  10.0]).tolist() #rope
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
# qddot_min[2] = -9.81
# qddot_max[2] = -9.81
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

L = 0.5*dot(qdot, qdot) # Objective term

tf = 1.0 #[s]

#Formulate discrete time dynamics
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
opts = {'tf': tf/ns}
F_integrator = integrator('F_integrator', 'rk', dae, opts)

# Start with an empty NLP

X, U = create_state_and_control([Q, Qdot], [Qddot, F1, F2, FRope])

V = concat_states_and_controls(X,U)

v_min, v_max = create_bounds([q_min, qdot_min], [q_max, qdot_max], [qddot_min, f_min1, f_min2, f_minRope], [qddot_max, f_max1, f_max2, f_maxRope], ns)

# Create Problem (J, v_min, v_max, g_min, g_max)

# Cost function
J = MX([0])
min_qdot = lambda k: 100.*dot(Qdot[k][6:-1], Qdot[k][6:-1])
J += cost_function(min_qdot, 0, ns)

min_qddot_a = lambda k: 1000.*dot(Qddot[k][6:-1], Qddot[k][6:-1])
J += cost_function(min_qddot_a, 0, ns-1)
min_F1 = lambda k: 1000.*dot(F1[k], F1[k])
J += cost_function(min_F1, 0, ns-1)
min_F2 = lambda k: 1000.*dot(F2[k], F2[k])
J += cost_function(min_F2, 0, ns-1)
min_FRope = lambda k: 100.*dot(FRope[k]-FRope[k-1], FRope[k]-FRope[k-1]) # min Fdot SUCA!
J += cost_function(min_FRope, 1, ns-1)

# dd = {'Contact1': F1, 'Contact2': F2, 'rope_anchor2': FRope}
dd = {'rope_anchor2': FRope}
id = inverse_dynamics(Q, Qdot, Qddot, ID, dd, kindyn)
Tau = id.compute_nodes(0, ns-1)
#
# min_torque = lambda k: 1.*dot(Tau[k], Tau[k])
# J += cost_function(min_torque, 0, ns-1)

# CONSTRAINTS
G = constraint_handler()

# Initial condition
x_init = q_init + qdot_init
init = initial_condition(X[0], x_init)
g1, g_min1, g_max1 = constraint(init, 0, 1)
G.set_constraint(g1, g_min1, g_max1)

# Multiple Shooting
multiple_shooting_constraint = multiple_shooting(X, Qddot, F_integrator)
g2, g_min2, g_max2 = constraint(multiple_shooting_constraint, 0, ns-1)
G.set_constraint(g2, g_min2, g_max2)

# Torque Limits
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

# inverse_dynamics



torque_lims1 = torque_lims(id, tau_min, tau_max)
g3, g_min3, g_max3 = constraint(torque_lims1, 0, ns-1)
G.set_constraint(g3, g_min3, g_max3)

# tau_min[15] = -10000.
# torque_lims2 = torque_lims(Jac_CRope, Q, Qdot, Qddot, FRope, ID, tau_min, tau_max)
# g4, g_min4, g_max4 = constraint(torque_lims2, 10, ns-1)
# G.set_constraint(g4, g_min4, g_max4)


# # Contact constraint
contact_constr = contact(FKRope, Q, q_init)
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


# PRINT AND REPLAY SOLUTION
dt = 0.05

solution_dict = retrieve_solution(V, {'Q': Q, 'Qdot': Qdot, 'Qddot': Qddot, 'F1': F1, 'F2': F2, 'FRope': FRope}, w_opt)

q_hist = solution_dict['Q']

#X_res, Qddot_res = resample_integrator(X, Qddot, tf, dt, dae)

#Resampler = Function("Resampler", [V], [X_res], ['V'], ['X_res'])

#x_hist_res = Resampler(V=w_opt)['X_res'].full()
#q_hist_res = (x_hist_res[0:nq,:]).transpose()
q_hist_res = q_hist


Resampler = Function("Resampler", [V], [Tau], ['V'], ['Tau'])
tau_hist = (Resampler(V=w_opt)['Tau'].full().flatten()).reshape(ns-1, nv)



# LOGGING
for k in solution_dict:
    logger.add(k, solution_dict[k])

logger.add('tau_hist', tau_hist)

logger.add('q_hist_res', q_hist_res)


del(logger)
#####

from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf as ros_tf
import geometry_msgs.msg

pub = rospy.Publisher('joint_states', JointState, queue_size=10)
rospy.init_node('joint_state_publisher')
rate = rospy.Rate(1./dt)
joint_state_pub = JointState()
joint_state_pub.header = Header()
joint_state_pub.name = ['Contact1_x', 'Contact1_y', 'Contact1_z',
                        'Contact2_x', 'Contact2_y', 'Contact2_z',
                        'rope_anchor1_1_x', 'rope_anchor1_2_y', 'rope_anchor1_3_z',
                        'rope_joint']

br = ros_tf.TransformBroadcaster()
m = geometry_msgs.msg.TransformStamped()
m.header.frame_id = 'world_odom'
m.child_frame_id = 'base_link'

while not rospy.is_shutdown():
    for k in range(ns):
        qk = q_hist_res[k]

        m.transform.translation.x = qk[0]
        m.transform.translation.y = qk[1]
        m.transform.translation.z = qk[2]
        quat = [qk[3], qk[4], qk[5], qk[6]]
        quat = normalize(quat)
        m.transform.rotation.x = quat[0]
        m.transform.rotation.y = quat[1]
        m.transform.rotation.z = quat[2]
        m.transform.rotation.w = quat[3]

        br.sendTransform((m.transform.translation.x, m.transform.translation.y, m.transform.translation.z),
                         (m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z,
                          m.transform.rotation.w),
                         rospy.Time.now(), m.child_frame_id, m.header.frame_id)

        joint_state_pub.header.stamp = rospy.Time.now()
        joint_state_pub.position = qk[7:nq]
        joint_state_pub.velocity = []
        joint_state_pub.effort = []
        pub.publish(joint_state_pub)
        rate.sleep()