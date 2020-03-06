from horizon import *
import casadi_kin_dyn.pycasadi_kin_dyn as cas_kin_dyn

class contact(constraint_class):
    def __init__(self, FKlink, Q, qinit):
        self.FKlink = FKlink
        self.Q = Q
        self.qinit = qinit

    def virtual_method(self, k):
        CLink_pos_init = self.FKlink(q=self.qinit)['ee_pos']
        CLink_pos = self.FKlink(q=self.Q[k])['ee_pos']
        self.gk = [CLink_pos - CLink_pos_init]
        self.g_mink = np.array([0., 0., 0.]).tolist()
        self.g_maxk = np.array([0., 0., 0.]).tolist()


class linearized_friction_cone(constraint_class):
    def __init__(self, Force, mu, Rot):
        self.F = Force
        self.mu = mu
        self.R = Rot

    def virtual_method(self, k):

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
        self.g_mink = np.array([-1000., -1000., -1000., -1000., -1000.]).tolist()
        self.g_maxk = np.array([0., 0., 0., 0., 0.]).tolist()

class remove_contact(constraint_class):
    def __init__(self, Force):
        self.F = Force

    def virtual_method(self, k):
        self.gk = [self.F[k]]
        self.g_mink = np.array([0., 0., 0.]).tolist()
        self.g_maxk = np.array([0., 0., 0.]).tolist()

class contact_handler(constraint_class):
    def __init__(self, FKlink, Force, number_of_nodes):
        self.FKlink = FKlink
        self.Force = Force
        self.ns = number_of_nodes

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

    def setFrictionCone(self, mu, Rot):
        self.friction_cone = linearized_friction_cone(self.Force, mu, Rot)

    def setContactAndFrictionCone(self, Q, q_contact, mu, Rot):
        self.setContact(Q, q_contact)
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
            if k < self.ns:
                self.friction_cone.virtual_method(k)
                self.g_fc, self.g_min_fc, self.g_max_fc = self.friction_cone.getConstraint()
        if self.remove_contact is not None:
            if k < self.ns:
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
                print "[WARNING] Friction Cones not set!"
        elif self.g_fc is not None:
            self.gk = self.g_fc
            self.g_mink = self.g_min_fc
            self.g_maxk = self.g_max_fc
            print "[WARNING] Contact not set!"
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




