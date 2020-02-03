#!/usr/bin/env python
from horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn
import matlogger2.matlogger as matl
import rospy

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

# Constraint
g = []
g_min = []
g_max = []

class force_unilaterality(constraint_class):
    def __init__(self, Jac_CRope, Q, Qdot, Qddot, F, ID):
        self.Q = Q
        self.Qdot = Qdot
        self.Qddot = Qddot
        self.F = F
        self.ID = ID
        self.Jac_CRope = Jac_CRope

    def virtual_method(self, k):
        CRope_jac = self.Jac_CRope(q=self.Q[k])['J']
        JtF = mtimes(CRope_jac.T, vertcat(self.F[k][6:9], MX.zeros(3, 1)))
        Tau = self.ID(q=self.Q[k], v=self.Qdot[k], a=self.Qddot[k])['tau'] - JtF

        self.gk = [Tau[15:16]]
        self.g_mink = np.array([-10000.]).tolist()
        self.g_maxk = np.array([0.]).tolist()

force_unilaterality_constraint = force_unilaterality(Jac_CRope, Q, Qdot, Qddot, F, ID)

dg, dg_min, dg_max = constraint(force_unilaterality_constraint, 0, ns)
g += dg
g_min += dg_min
g_max += dg_max

gg = []
gg_min = []
gg_max = []
for k in range(ns):
    CRope_jac = Jac_CRope(q=Q[k])['J']
    JtF = mtimes(CRope_jac.T, vertcat(F[k][6:9], MX.zeros(3,1)))

    Tau = ID(q=Q[k], v=Qdot[k], a=Qddot[k])['tau'] - JtF





    # # Multiple shooting constraint
    # integrator_out = F_integrator(x0=X[k], p=Qddot[k])
    # g += [integrator_out['xf'] - X[k + 1]]
    # g_min += [0] * X[k + 1].size1()
    # g_max += [0] * X[k + 1].size1()
    #
    # # Underactuation on spherical joint rope
    # g += [Tau[12:15]]
    # g_min += np.array([0., 0., 0.]).tolist()
    # g_max += np.array([0., 0., 0.]).tolist()
    #
    # # Floating base constraint
    # g += [Tau[0:6]]
    # g_min += np.zeros((6, 1)).tolist()
    # g_max += np.zeros((6, 1)).tolist()
    #
    # # Contact constraints
    # CRope_pos_init = FKRope(q=q_init)['ee_pos']
    # CRope_pos = FKRope(q=Q[k])['ee_pos']
    # g += [CRope_pos - CRope_pos_init]
    # g_min += np.array([0., 0., 0.]).tolist()
    # g_max += np.array([0., 0., 0.]).tolist()

    # Unilaterality on rope Force
    gg += [Tau[15:16]]
    gg_min += np.array([-10000.]).tolist()
    gg_max += np.array([0.]).tolist()



print "g_min: ", g_min
print "gg_min: ", gg_min










