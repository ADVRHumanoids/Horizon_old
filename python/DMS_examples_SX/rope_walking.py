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

logger = []
logger = matl.MatLogger2('/tmp/rope_walking_log')
logger.setBufferMode(matl.BufferMode.CircularBuffer)

urdf = rospy.get_param('robot_description')
kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

FKR = Function.deserialize(kindyn.fk('Contact1'))
FKL = Function.deserialize(kindyn.fk('Contact2'))

# Inverse Dynamics
ID = Function.deserialize(kindyn.rnea())

# OPTIMIZATION PARAMETERS
ns = 80  # number of shooting nodes

nc = 3  # number of contacts

nq = kindyn.nq()  # number of DoFs - NB: 7 DoFs floating base (quaternions)

DoF = nq - 7  # Contacts + anchor_rope + rope

nv = kindyn.nv()  # Velocity DoFs

nf = 3  # 2 feet contacts + rope contact with wall, Force DOfs

# VARIABLES
f1, F1 = create_variable('F1', nf, ns, 'CONTROL', 'SX')
f_min1 = (-10000.*np.ones(nf)).tolist()
f_max1 = (10000.*np.ones(nf)).tolist()
f_init1 = np.zeros(nf).tolist()

f2, F2 = create_variable('F2', nf, ns, 'CONTROL', 'SX')
f_min2 = (-10000.*np.ones(nf)).tolist()
f_max2 = (10000.*np.ones(nf)).tolist()
f_init2 = np.zeros(nf).tolist()


#FOOSTEP SCHEDULER
# start_node = 0
# walking_phases = 4
# nodes_per_action = 5
actions_dict = collections.OrderedDict()
# actions_dict['L'] = []
# actions_dict['R'] = []
# actions_dict['D'] = []
# actions_dict['N'] = []
# footsep_scheduler = footsteps_scheduler(start_node, walking_phases, nodes_per_action, ns, actions_dict)

start_node = 0
flying_phases = 2
nodes_per_action = 3

N_action_1 = cons.contact.contact_handler(FKR, F1)
N_action_1.removeContact()
N_action_2 = cons.contact.contact_handler(FKL, F2)
N_action_2.removeContact()
actions_dict['N'] = [N_action_1, N_action_2]
footsep_scheduler = footsteps_scheduler(start_node, flying_phases, nodes_per_action, ns, actions_dict)
g, gmin, gmax = footsep_scheduler.get_constraints()

G = constraint_handler()
G.set_constraint([g], gmin, gmax)
g, gmin, gmax = G.get_constraints()

print "g: ", g, "g len: ", g.size()
print "gmin: ", gmin, "gmin len: ", len(gmin)
print "gmax: ", gmax, "gmax len: ", len(gmax)