from casadi import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

def plot_solution(q):

    def quaternion_to_euler(quat):
        n_res = np.shape(quat)[0]
        euler = np.ones((n_res, 3))
        for k in range(int(round(n_res))):
            r = Rot.from_quat(quat[k])
            euler[k] = r.as_euler('xyz')

        return euler

    plt.subplot(211)
    plt.plot(q[:, 0:3])
    plt.xlabel('$\mathrm{node}$')
    plt.ylabel('$\mathrm{Position}$')
    plt.grid()
    plt.subplot(212)
    plt.plot(quaternion_to_euler(q[:, 3:7]))
    plt.xlabel('$\mathrm{node}$')
    plt.ylabel('$\mathrm{Orientation} \quad [XYZ]$')
    plt.grid()
    plt.suptitle('$\mathrm{Floating Base}$')
    plt.show()
