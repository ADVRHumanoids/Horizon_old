#!/usr/bin/env python
from horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn
import matlogger2.matlogger as matl
import rospy
from constraints import *

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
nf = 3*nc # 2 feet contacts + rope contact with wall, Force DOfs

# Variables
q, Q = create_variable("Q", nq, ns, "STATE")
q_min = np.array([-10.0, -10.0, -10.0, -1.0, -1.0, -1.0, -1.0, # Floating base
                  -0.3, -0.1, -0.1, # Contact 1
                  -0.3, -0.05, -0.1, # Contact 2
                  -1.57, -1.57, -3.1415, #rope_anchor
                  0.5]) #rope
q_max = np.array([10.0,  10.0,  10.0,  1.0,  1.0,  1.0,  1.0, # Floating base
                  0.3, 0.05,  0.1, # Contact 1
                  0.3, 0.1,  0.1, # Contact 2
                  1.57, 1.57, 3.1415,  #rope_anchor
                  0.5]) #rope
q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                   0., 0., 0.,
                   0., 0., 0.,
                   0., 0., 0.,
                   0.5])

qdot, Qdot = create_variable('Qdot', nv, ns, "STATE")
qdot_min = np.full((1, nv), -100.)
qdot_max = np.full((1, nv), 100.)

qddot, Qddot = create_variable('Qddot', nv, ns, "CONTROL")
qddot_min = np.full((1, nv), -100.)
qddot_max = np.full((1, nv), 100.)


f, F = create_variable('F', nf, ns, "CONTROL")
f_min = np.tile(np.array([-10000., -10000., -10000.]), nc)
f_max = np.tile(np.array([10000., 10000., 10000.]), nc)


x, xdot = dynamic_model_with_floating_base(q, qdot, qddot)

L = 0.01*dot(qddot, qddot) # Objective term

tf = 3. #[s]

#Formulate discrete time dynamics
dae = {'x': x, 'p': qddot, 'ode': xdot, 'quad': L}
opts = {'tf': tf/ns}
F_integrator = integrator('F_integrator', 'rk', dae, opts)

# Start with an empty NLP

X, U = create_state_and_control([Q, Qdot], [Qddot, F])
V = concat_states_and_controls(X,U)

v_min, v_max = create_bounds([q_min, qdot_min], [q_max, qdot_max], [qddot_min, f_min], [qddot_max, f_max], ns)

# Create Problem (J, v_min, v_max, g_min, g_max)

# Cost function
J = MX([0])
min_qdot = lambda k: 1*dot(Qdot[k], Qdot[k])
J += cost_function(min_qdot, 0, ns)

# CONSTRAINTS
g = []
g_min = []
g_max = []

# Multiple Shooting
multiple_shooting_constraint = multiple_shooting(X, Qddot, F_integrator)
dg, dg_min, dg_max = constraint(multiple_shooting_constraint, 0, ns)
g += dg
g_min += dg_min
g_max += dg_max

# Torque Limits
# 0-5 Floating base constraint
tau_min = np.zeros((6, 1)).tolist()
tau_max = np.zeros((6, 1)).tolist()
# 6-11 Actuated Joints, free
tau_min += np.full((6,1), -1000.).tolist()
tau_max += np.full((6,1),  1000.).tolist()
# 12-14 Underactuation on spherical joint rope
tau_min += np.array([0., 0., 0.]).tolist()
tau_max += np.array([0., 0., 0.]).tolist()
# 15 force rope unilaterality
tau_min += np.array([-10000.]).tolist()
tau_max += np.array([0.]).tolist()

torque_lims = torque_lims(Jac_CRope, Q, Qdot, Qddot, F, ID, tau_min, tau_max)
dg, dg_min, dg_max = constraint(torque_lims, 0, ns)
g += dg
g_min += dg_min
g_max += dg_max

# Contact constraint
contact_constr = contact(FKRope, Q, q_init)

dg, dg_min, dg_max = constraint(contact_constr, 0, ns)
g += dg
g_min += dg_min
g_max += dg_max
















