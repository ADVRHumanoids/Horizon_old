from Horizon.horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn
from enum import Enum

class contact_type(Enum):
    """
    Enumeration class used to identify the type of contact
    """
    point = 0
    flat = 1

class contact(constraint_class):
    """
    Position constraint on a desired link given a joint space desired position
    """
    def __init__(self, FKlink, Q, qinit):
        """
        Constructor
        Args:
            FKlink: forward kinematics function
            Q: position state variables
            qinit: desired joint space variables to evaluate in forwad kineamtics
        """
        self.FKlink = FKlink
        self.Q = Q
        self.qinit = qinit

    def virtual_method(self, k):
        """
        Compute constraint at given node
        Args:
            k: node
        """
        CLink_pos_init = self.FKlink(q=self.qinit)['ee_pos']
        CLink_pos = self.FKlink(q=self.Q[k])['ee_pos']
        self.gk = [CLink_pos - CLink_pos_init]
        self.g_mink = np.array([0., 0., 0.]).tolist()
        self.g_maxk = np.array([0., 0., 0.]).tolist()

class surface_contact(constraint_class):
    """
    Position constraint to lies into a plane: ax + by + cz +d = 0 together with 0 Cartesian velocity of the contact
    """
    def __init__(self, plane_dict, FKlink, Q, Jac, Qdot, contact_type):
        """
        Constructor
        Args:
            plane_dict: which contains following variables:
                a
                b
                c
                d
                to define the plane ax + by + cz +d = 0
            FKlink: forward kinematics function of desired link
            Q: position state variables
            Jac: Jacobian function of the link
            Qdot: velocity state variables
        """
        self.P = np.array([0., 0., 0.])
        self.d = 0.

        if 'a' in plane_dict:
            self.P[0] = plane_dict['a']
        if 'b' in plane_dict:
            self.P[1] = plane_dict['b']
        if 'c' in plane_dict:
            self.P[2] = plane_dict['c']

        if 'd' in plane_dict:
            self.d = plane_dict['d']

        self.FKlink = FKlink
        self.Jac = Jac
        self.Q = Q
        self.Qdot = Qdot
        self.__contact_type = contact_type

    def virtual_method(self, k):
        """
            Compute constraint at given node
            Args:
                k: node
        """
        CLink_pos = self.FKlink(q=self.Q[k])['ee_pos']
        CLink_jac = self.Jac(q=self.Q[k])['J']

        if self.__contact_type is contact_type.point:
            self.gk = [dot(self.P, CLink_pos), mtimes(CLink_jac[0:3,:], self.Qdot[k])]
            self.g_mink = np.array([-self.d, 0.0, 0.0, 0.0]).tolist()
            self.g_maxk = np.array([-self.d, 0.0, 0.0, 0.0]).tolist()
        elif self.__contact_type is contact_type.flat:
            self.gk = [dot(self.P, CLink_pos), mtimes(CLink_jac, self.Qdot[k])]
            self.g_mink = np.array([-self.d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).tolist()
            self.g_maxk = np.array([-self.d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).tolist()
        else:
            raise ValueError('Specified contact type not implemented!')

class surface_contact_gap(constraint_class):
    """
    TODO: consider a gap not only in z
    Position constraint to lies into a plane: ax + by + cz +d = 0 with a gap, together with 0 Cartesian velocity of
    the contact. The gap is modeled as:
    (z-gap_min)w(z-gap_max) >= 0
    """
    def __init__(self, plane_dict, FKlink, Q, Jac, Qdot):
        """
        Constructor
        Args:
            plane_dict: which contains following variables:
                a
                b
                c
                d
                    to define the plane ax + by + cz +d = 0
                z_gap_min: min z coordinate of the gap
                z_gap_max: max z coordinate of the gap
                relaxation: gain to relax the constraint of the gap
            FKlink: forward kinematics function of desired link
            Q: position state variables
            Jac: Jacobian function of the link
            Qdot: velocity state variables
        """
        self.P = np.array([0., 0., 0.])
        self.d = 0.

        if 'a' in plane_dict:
            self.P[0] = plane_dict['a']
        if 'b' in plane_dict:
            self.P[1] = plane_dict['b']
        if 'c' in plane_dict:
            self.P[2] = plane_dict['c']

        if 'd' in plane_dict:
            self.d = plane_dict['d']

        if 'z_gap_min' in plane_dict:
            self.z_gap_min = plane_dict['z_gap_min']

        if 'z_gap_max' in plane_dict:
            self.z_gap_max = plane_dict['z_gap_max']

        self.FKlink = FKlink
        self.Jac = Jac
        self.Q = Q
        self.Qdot = Qdot

        self.__w = 1.
        if 'relaxation' in plane_dict:
            self.__w = plane_dict['relaxation']

    def virtual_method(self, k):
        """
            Compute constraint at given node
            Args:
                k: node
        """
        CLink_pos = self.FKlink(q=self.Q[k])['ee_pos']
        CLink_jac = self.Jac(q=self.Q[k])['J'] #TODO: give possibility to lock position and/or orientation

        # GAP: inequality
        self.gk = [dot(self.P, CLink_pos), mtimes(CLink_jac[0:3, :], self.Qdot[k]), (CLink_pos[2]-self.z_gap_max)*self.__w*(CLink_pos[2]-self.z_gap_min)]
        self.g_mink = np.array([-self.d, 0.0, 0.0, 0.0, 0.0]).tolist()
        self.g_maxk = np.array([-self.d, 0.0, 0.0, 0.0, 10000.0]).tolist()




class linearized_friction_cone(constraint_class):
    """
    Friction cone constraint
    """
    def __init__(self, Force, mu, Rot):
        """
        Constructor
        Args:
            Force: control variable
            mu: friction coefficient
            Rot: rotation matrix associated to friction cone

        TODO: Add constructor for normal/plane input
        """
        self.F = Force
        self.mu = mu
        self.R = Rot

    def virtual_method(self, k):
        """
            Compute constraint at given node
            Args:
                k: node
        """

        mu_lin = self.mu / 2.0 * sqrt(2.0)

        A_fr = np.zeros([5, 3])
        A_fr[0, 0] = 1.0
        A_fr[0, 2] = -mu_lin
        A_fr[1, 0] = -1.0
        A_fr[1, 2] = -mu_lin
        A_fr[2, 1] = 1.0
        A_fr[2, 2] = -mu_lin
        A_fr[3, 1] = -1.0
        A_fr[3, 2] = -mu_lin
        A_fr[4, 2] = -1.0

        A_fr_R = mtimes(A_fr, self.R)

        self.gk = [mtimes(A_fr_R, self.F[k])]
        self.g_mink = np.array([-10000., -10000., -10000., -10000., -10000.]).tolist()
        self.g_maxk = np.array([0., 0., 0., 0., 0.]).tolist()

class remove_contact(constraint_class):
    """
    Class which imposes zero forces to a contact forces
    """
    def __init__(self, Force):
        """
        Constructor
        Args:
            Force: control variable
        """
        self.F = Force

    def virtual_method(self, k):
        """
            Compute constraint at given node
            Args:
                k: node
        """

        self.gk = [self.F[k]]
        self.g_mink = np.array([0., 0., 0.]).tolist()
        self.g_maxk = np.array([0., 0., 0.]).tolist()

class contact_handler(constraint_class):
    """
    Class to handle contact switching (kinematic part and force)
    """
    def __init__(self, FKlink, Force):
        """
        Constructor
        Args:
            FKlink: forward kinematics of link in contact
            Force: force control variable associated to link in contact
        """
        self.FKlink = FKlink

        self.Force = Force
        self.cns = np.size(Force)

        self.kinematic_contact = None
        self.friction_cone = None
        self.remove_contact = None

        self.g_kc = None
        self.g_max_kc = None
        self.g_min_kc = None

        self.g_fc = None
        self.g_max_fc = None
        self.g_min_fc = None

        self.g_nc = None
        self.g_max_nc = None
        self.g_min_nc = None

    def setContact(self, Q, q_contact):
        self.kinematic_contact = contact(self.FKlink, Q, q_contact)

    def setSurfaceContact(self, plane_dict, Q, Jac, Qdot, contact_type):
        self.kinematic_contact = surface_contact(plane_dict, self.FKlink, Q, Jac, Qdot, contact_type)

    def setSurfaceContactGap(self, plane_dict, Q, Jac, Qdot):
        self.kinematic_contact = surface_contact_gap(plane_dict, self.FKlink, Q, Jac, Qdot)

    def setFrictionCone(self, mu, Rot):
        self.friction_cone = linearized_friction_cone(self.Force, mu, Rot)

    def setContactAndFrictionCone(self, Q, q_contact, mu, Rot):
        self.setContact(Q, q_contact)
        self.setFrictionCone(mu, Rot)

    def setSurfaceContactAndFrictionCone(self, Q, plane_dict, Jac, Qdot, contact_type, mu, Rot): #TODO: remove Rot, can be extracted from plane_dict!
        self.setSurfaceContact(plane_dict, Q, Jac, Qdot, contact_type)
        self.setFrictionCone(mu, Rot)

    def setSurfaceContactGapAndFrictionCone(self, Q, plane_dict, Jac, Qdot, mu, Rot): #TODO: remove Rot, can be extracted from plane_dict!
        self.setSurfaceContactGap(plane_dict, Q, Jac, Qdot)
        self.setFrictionCone(mu, Rot)

    def removeContact(self):
        self.friction_cone = None
        self.kinematic_contact = None
        self.remove_contact = remove_contact(self.Force)

    def virtual_method(self, k):
        if self.kinematic_contact is not None:
            self.kinematic_contact.virtual_method(k)
            self.g_kc, self.g_min_kc, self.g_max_kc = self.kinematic_contact.getConstraint()
        if self.friction_cone is not None:
            if k < self.cns:
                self.friction_cone.virtual_method(k)
                self.g_fc, self.g_min_fc, self.g_max_fc = self.friction_cone.getConstraint()
        if self.remove_contact is not None:
            if k < self.cns:
                self.remove_contact.virtual_method(k)
                self.g_nc, self.g_min_nc, self.g_max_nc = self.remove_contact.getConstraint()

        if self.g_kc is not None:
            if self.g_fc is not None:
                self.gk = self.g_kc + self.g_fc
                self.g_mink = self.g_min_kc + self.g_min_fc
                self.g_maxk = self.g_max_kc + self.g_max_fc
            else:
                self.gk = self.g_kc
                self.g_mink = self.g_min_kc
                self.g_maxk = self.g_max_kc
        elif self.g_fc is not None:
            self.gk = self.g_fc
            self.g_mink = self.g_min_fc
            self.g_maxk = self.g_max_fc
        elif self.g_nc is not None:
            self.gk = self.g_nc
            self.g_mink = self.g_min_nc
            self.g_maxk = self.g_max_nc
        else:
            raise ValueError('Neither Contact nor Friction Cone have been set!')

        self.g_kc = None
        self.g_max_kc = None
        self.g_min_kc = None

        self.g_fc = None
        self.g_max_fc = None
        self.g_min_fc = None

        self.g_nc = None
        self.g_max_nc = None
        self.g_min_nc = None




