from casadi import *

def normalize(v):
    return v / np.linalg.norm(v)

def normalize_quaternion(q):

    n_res = np.shape(q)[0]
    for k in range(int(round(n_res))):
        qk = q[k]
        quat = normalize([qk[3], qk[4], qk[5], qk[6]])
        q[k][3:7] = quat[0:4]

    return q