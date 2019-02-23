import numpy as np
from numpy.linalg import norm

##### quaternion to rotation vector ##### ?????
def quat2vec(quat):
    """
    :param quat: (4x2n)
    :return vec: (3x2n)
    """

    # # avoid changing original matrix
    # q = quat.copy()
    #
    # theta = np.arccos(q[0]) * 2
    # sin = np.sin(theta / 2)
    # vec = (theta / sin) * q[1:,:]
    # vec[np.isnan(vec)] = 0
    # vec[np.isinf(vec)] = 0

    vec = quat[1:,:].copy()

    return vec

##### rotation vector to quaternion #####
def vec2quat(vec):
    """
    :param vec: (3x2n)
    :return quat: (4x2n)
    """
    theta = norm(vec, axis=0)
    axis = np.divide(vec, theta)
    quat = np.zeros((4,vec.shape[1]))
    quat[0] = np.cos(theta/2.)
    quat[1:] = axis * np.sin(theta/2.)

    # if norm == 0 would cause NaN or Inf
    quat[np.isnan(quat)] = 0
    quat[np.isinf(quat)] = 0

    return quat

##### quaternion to roll, pitch, yaw #####
def quat2rpy(quat):
    """
    :param quat: (4x12)
    :return row, pitch, yaw:
    """
    q = quat.copy()

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # roll (x-axis)
    r_sin = 2 * (qw * qx + qy * qz)
    r_cos = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(r_sin, r_cos)

    # pitch (y-axis)
    p_sin = 2 * (qw * qy - qz * qx)
    pitch = np.arcsin(p_sin)
    # if p_sin.any() >= 1:
    #     print('p_sin>=1')

    # yaw (z-axis)
    y_sin = 2 * (qw * qz + qx * qy)
    y_cos = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(y_sin, y_cos)

    return roll, pitch, yaw

##### rotation matrix to roll, pitch, yaw #####
def rot2rpy(rot):
    """
    :param rot: (3,3,n)
    :return rpy: (3,n)
    """
    rpy = np.zeros((3, rot.shape[2]))

    rpy[0] = np.arctan2(rot[2, 1, :], rot[2, 2, :])
    rpy[1] = np.arctan2(-rot[2, 0, :], np.sqrt(rot[2, 1, :] ** 2 + rot[2, 2, :] ** 2))
    rpy[2] = np.arctan2(rot[1, 0, :], rot[0, 0, :])

    return rpy

##### angular velocity to quaternion #####
def omg2quat(omg, t):
    """
    :param omg:
    :param t:
    :return:
    """
    w_norm = norm(omg)
    if w_norm == 0:
        return np.asarray((1,0,0,0)).reshape((-1,1))

    theta = w_norm * t
    axis = omg / w_norm.astype(float)

    quat = np.zeros((4,1))
    quat[0] = np.cos(theta/2.)
    quat[1:] = axis * np.sin(theta/2.)

    return quat

##### quaternion multiplication #####
def quatMulti(quat0, quat1):
    """
    :param quat0: (4,m)
    :param quat1: (4,n)
    :return quat: (4,n)
    """
    # reshape
    quat0 = quat0.reshape((4,-1))
    quat1 = quat1.reshape((4, -1))

    w0, x0, y0, z0 = quat0[0], quat0[1], quat0[2], quat0[3]
    w1, x1, y1, z1 = quat1[0], quat1[1], quat1[2], quat1[3]

    quat = np.zeros((4,max(quat0.shape[1],quat1.shape[1])))

    # multiplication
    quat[0] = np.multiply(w0,w1) - np.multiply(x0,x1) \
              - np.multiply(y0,y1) - np.multiply(z0,z1)

    quat[1] = np.multiply(w0,x1) + np.multiply(x0,w1) \
              + np.multiply(y0,z1) - np.multiply(z0,y1)

    quat[2] = np.multiply(w0,y1) - np.multiply(x0,z1) \
              + np.multiply(y0,w1) + np.multiply(z0,x1)

    quat[3] = np.multiply(w0,z1) + np.multiply(x0,y1) \
              - np.multiply(y0,x1) + np.multiply(z0,w1)

    # normalize
    quat = quat / norm(quat, axis=0)

    return quat


def quatMean(quat):
    """
    :param quat: (4x12)
    :return quat_avg: (4,1)
    """
    # form the symmetric accumulator matrix
    A = np.zeros((4,4))

    for i in range(12):
        A = np.dot(quat[:,[i]], np.transpose(quat[:,[i]])) + A

    # scale
    A = (1/12.) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    idx = np.argmax(eigenValues)
    # return the real part of the largest eigenvector (has only real part)
    quat_avg = eigenVectors[:, [idx]]

    return quat_avg.reshape((-1,1))


def quatInv(quat):
    """
    :param quat: (4x2n)
    :return quat_inv: (4x2n)
    """
    quat_inv = quat.copy()
    quat_inv[1:,:] *= -1

    return quat_inv

def getCov(rot):
    """
    :param rot: (6x2n)
    :return cov: (6x6)
    """
    n = rot.shape[1]
    cov = np.dot(rot, np.transpose(rot))/n

    return cov

def getMean(rot):

    n = rot.shape[1]
    row_sum = rot.sum(axis=1).reshape((-1,1))
    mean = row_sum/n

    return mean

if __name__ == "__main__":

    q_1 = np.array([[0.707, 0, 0, 0.707], [0, 0.707, 0.707, 0]])
    q_2 = np.array([[1, 0, 0, 0]])
    q_res = quatMulti(q_1.T, q_2.T)
    print(q_res)
    print(quat2vec(q_res))
    print(vec2quat(quat2vec(q_res)))