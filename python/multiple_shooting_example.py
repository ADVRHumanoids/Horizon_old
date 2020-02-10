#!/usr/bin/env python
from horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn
import matlogger2.matlogger as matl
import rospy
from constraints import *
from previewer import *
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
print "ns: ", ns

nc = 3  # number of contacts
print "nc: ", nc

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)
print "nq:", nq

DoF = nq - 7 # Contacts + anchor_rope + rope
print "DoF: ", DoF

nv = kindyn.nv() # Velocity DoFs
print "nv: ", nv

nf = 3*nc # 2 feet contacts + rope contact with wall, Force DOfs
print "nf: ", nf

# Variables
q, Q = create_variable("Q", nq, ns, "STATE")
q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0, # Floating base
                  -0.3, -0.1, -0.1, # Contact 1
                  -0.3, -0.05, -0.1, # Contact 2
                  -1.57, -1.57, -3.1415, #rope_anchor
                  0.0]).tolist() #rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0, # Floating base
                  0.3, 0.05,  0.1, # Contact 1
                  0.3, 0.1,  0.1, # Contact 2
                  1.57, 1.57, 3.1415,  #rope_anchor
                  0.5]).tolist() #rope
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0.,
                   0.1]).tolist()

print "Q: ", Q
print "Q size: ", np.size(Q)
print "q_min: ", q_min
print "q_min size: ", np.size(q_min)
print "q_max: ", q_max
print "q_max size: ", np.size(q_max)
print "q_init: ", q_init
print "q_init size: ", np.size(q_init)

qdot, Qdot = create_variable('Qdot', nv, ns, "STATE")
qdot_min = (-100.*np.ones(nv)).tolist()
qdot_max = (100.*np.ones(nv)).tolist()
qdot_init = np.zeros(nv).tolist()

print "Qdot:", Qdot
print "Qdot size: ", np.size(Qdot)
print "qdot_min: ", qdot_min
print "qdot_min size: ", np.size(qdot_min)
print "qdot_max: ", qdot_max
print "qdot_max size: ", np.size(qdot_max)
print "qdot_init: ", qdot_init
print "qdot_init size: ", np.size(qdot_init)

qddot, Qddot = create_variable('Qddot', nv, ns, "CONTROL")
qddot_min = (-100.*np.ones(nv)).tolist()
qddot_max = (100.*np.ones(nv)).tolist()
qddot_init = np.zeros(nv).tolist()

print "Qddot:", Qddot
print "Qddot size: ", np.size(Qddot)
print "qddot_min: ", qddot_min
print "qddot_min size: ", np.size(qddot_min)
print "qddot_max: ", qddot_max
print "qddot_max size: ", np.size(qddot_max)
print "qddot_init: ", qddot_init
print "qddot_init size: ", np.size(qddot_init)

f, F = create_variable('F', nf, ns, "CONTROL")
f_min = (-10000.*np.ones(nf)).tolist()
f_max = (10000.*np.ones(nf)).tolist()
f_init = np.zeros(nf).tolist()

print "F:", F
print "F size: ", np.size(F)
print "f_min: ", f_min
print "f_min size: ", np.size(f_min)
print "f_max: ", f_max
print "f_max size: ", np.size(f_max)
print "f_init: ", f_init
print "f_init size: ", np.size(f_init)


x, xdot = dynamic_model_with_floating_base(q, qdot, qddot)

L = 0.01*dot(qddot, qddot) # Objective term

tf = 1. #[s]

#Formulate discrete time dynamics
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
opts = {'tf': tf/ns}
F_integrator = integrator('F_integrator', 'rk', dae, opts)

# Start with an empty NLP

X, U = create_state_and_control([Q, Qdot], [Qddot, F])
print "X:", X
print "X size: ", np.size(X)
print "U:", U
print "U size: ", np.size(U)

V = concat_states_and_controls(X,U)
print "V:", V
print "V size: ", V.size1()

v_min, v_max = create_bounds([q_min, qdot_min], [q_max, qdot_max], [qddot_min, f_min], [qddot_max, f_max], ns)
print "v_min: ", v_min
print "v_min size: ", v_min.size1()
print "v_max: ", v_max
print "v_max size: ", v_max.size1()

# Create Problem (J, v_min, v_max, g_min, g_max)

# Cost function
J = MX([0])
min_qdot = lambda k: 100.*dot(Qdot[k], Qdot[k])
J += cost_function(min_qdot, 0, ns)

# CONSTRAINTS
g = []
g_min = []
g_max = []

# Initial condition
init = initial_condition(Q, q_init)
dg, dg_min, dg_max = constraint(init, 0, 1)
g += dg
g_min += dg_min
g_max += dg_max


# Multiple Shooting
multiple_shooting_constraint = multiple_shooting(X, Qddot, F_integrator)
dg, dg_min, dg_max = constraint(multiple_shooting_constraint, 0, ns-1)
g += dg
g_min += dg_min
g_max += dg_max


# # Torque Limits
# # 0-5 Floating base constraint
tau_min = np.zeros((6, 1)).tolist()
tau_max = np.zeros((6, 1)).tolist()
# # # 6-11 Actuated Joints, free
tau_min += np.full((6,1), -1000.).tolist()
tau_max += np.full((6,1),  1000.).tolist()
# # # 12-14 Underactuation on spherical joint rope
tau_min += np.array([0., 0., 0.]).tolist()
tau_max += np.array([0., 0., 0.]).tolist()
# # # 15 force rope unilaterality
#tau_min += np.array([-10000.]).tolist()
tau_min += np.array([0.]).tolist()
tau_max += np.array([0.]).tolist()
#
torque_lims1 = torque_lims(Jac_CRope, Q, Qdot, Qddot, F, ID, tau_min, tau_max)
dg, dg_min, dg_max = constraint(torque_lims1, 0, 10)
g += dg
g_min += dg_min
g_max += dg_max

tau_min[15] = np.array([-10000.]).tolist()
torque_lims2 = torque_lims(Jac_CRope, Q, Qdot, Qddot, F, ID, tau_min, tau_max)
dg, dg_min, dg_max = constraint(torque_lims2, 10, ns-1)
g += dg
g_min += dg_min
g_max += dg_max
#

# # Contact constraint
contact_constr = contact(FKRope, Q, q_init)
dg, dg_min, dg_max = constraint(contact_constr, 0, ns)
g += dg
g_min += dg_min
g_max += dg_max



opts = {'ipopt.tol': 1e-3,
        'ipopt.max_iter': 2000,
        'ipopt.linear_solver': 'ma57'}

solver = nlpsol('solver', 'ipopt', {'f': J, 'x': V, 'g': vertcat(*g)}, opts)

x0 = create_init([q_init, qdot_init], [qddot_init, f_init], ns)


sol = solver(x0=x0, lbx=v_min, ubx=v_max, lbg=vertcat(*g_min), ubg=vertcat(*g_max))
w_opt = sol['x'].full().flatten()


# PRINT AND REPLAY SOLUTION
dt = 0.01
#q_hist_res = trajectory_resampler(ns, F_integrator, V, X, Qddot, tf, dt, nq, nq+nv, w_opt)





resampler = Function("Resampler", [V], [vertcat(*Q)] ,['V'], ['Q'])
q_hist = resampler(V=w_opt)['Q'].full()

q_hist_res = q_hist.reshape(ns, nq)
print "q_hist_res: ", q_hist_res

from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import tf as ros_tf
import geometry_msgs.msg

pub = rospy.Publisher('joint_states', JointState, queue_size=10)
rospy.init_node('joint_state_publisher')
rate = rospy.Rate(1. / dt)
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
        m.transform.translation.x = q_hist_res[k, 0]
        m.transform.translation.y = q_hist_res[k, 1]
        m.transform.translation.z = q_hist_res[k, 2]
        quat = [q_hist_res[k, 3],q_hist_res[k, 4],q_hist_res[k, 5],q_hist_res[k, 6]]
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
        joint_state_pub.position = q_hist_res[k, 7:nq]
        joint_state_pub.velocity = []
        joint_state_pub.effort = []
        pub.publish(joint_state_pub)
        rate.sleep()











