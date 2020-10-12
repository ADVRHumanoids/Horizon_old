from casadi import *
import math
from scipy.spatial.transform import Rotation as Rot


def rotation_matrix_to_euler(R):

    def rotationMatrixToEulerAngles(R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    n_res = np.shape(R)[0]
    euler = np.ones((n_res, 3))
    for k in range(int(round(n_res))):
        euler[k] = rotationMatrixToEulerAngles(R[k])

    return euler

def quaternion_to_euler(quat):
    n_res = np.shape(quat)[0]
    euler = np.ones((n_res, 3))
    for k in range(int(round(n_res))):
        r = Rot.from_quat(quat[k])
        euler[k] = r.as_euler('xyz')

    return euler