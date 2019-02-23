import numpy as np
from quaternion import *
from scipy.linalg import cholesky

###### get X_i from W_i #####
def sigmaPoints(P_k, x_hat_k, Q):
    """
    :param P_k: (6x6) Estimate error covariance
    :param x_hat_k: (7x1) Posterior estimate
    :param Q: (6x6) Process noise covariace
    :return quat_k_bar: X_i set
    :return omg_k_bar: X_i set
    """

    # step1: W_i set distributed around zero with covariance (6x12)
    s = cholesky(P_k + Q)
    pq = np.sqrt(12) * s
    W_i = np.hstack((pq, -pq))

    # step2: W_i set to X_i set (7x12) add noise
    vec_w = W_i[:3, :]
    omg_w = W_i[3:, :]
    quat_w = vec2quat(vec_w)

    # previous state quaternion and angular velocity
    quat_k = x_hat_k[:4,[0]]
    omg_k = x_hat_k[4:,[0]]

    quat_k_bar = quatMulti(quat_k, quat_w)
    omg_k_bar = omg_k + omg_w

    return quat_k_bar, omg_k_bar

##### get Y_i from X_i #####
def processModel(quat_k_bar, omg_k_bar, t_delta):
    """
    :param quat_k_bar: (4x12) quaternion sigma points (X_i set)
    :param omg_k_bar: (3x12) angular velocity sigma points (X_i set)
    :param vals_imu: (6x1) imu data
    :param ts_imu: time duration
    :return quat_k:
    """

    # step3: X_i set to Y_i set (7x2n) process model
    omg_delta = omg_k_bar * t_delta
    q_delta = vec2quat(omg_delta)

    """
    omg_mean = norm(x_hat_k[3:])
    if omg_mean == 0:
        q_delta = np.asarray([[1],[0],[0],[0]])
    else:
        theta = omg_mean * t_delta
        axis = x_hat_k[3:] / omg_mean
        q_delta = np.vstack((np.cos(theta/2.), axis * np.sin(theta/2.)))
    """

    quat_k = quatMulti(quat_k_bar, q_delta)

    return quat_k, omg_k_bar

##### get the prior estimate #####
def prediction(quat_k, omg_k):
    """
    :param quat_k: (4x12) Y_i set
    :param omg_k: (3x12) Y_i set
    :return P_k_bar: (6x6) prior covariance
    :return x_hat_k_bar: (7x1) prior estimate
    :return W_i_prime:
    """

    # step4: compute prior estimate x_hat_k_bar from the mean of the sigma points
    quat_mean = quatMean(quat_k)
    omg_mean = getMean(omg_k)

    W_i_prime = np.zeros((6, 12))
    W_i_prime[:3, :] = quat2vec(quatMulti(quat_k, quatInv(quat_mean)))
    W_i_prime[3:, :] = omg_k - omg_mean

    x_hat_k_bar = np.vstack((quat_mean, omg_mean))

    # step5: get prior process covariance P_k_bar
    P_k_bar = getCov(W_i_prime)

    return P_k_bar, x_hat_k_bar, W_i_prime

##### get Z_i from Y_i #####
def measurementModel(quat_k, omg_k):
    """
    :param quat_k: (4x12) Y_i set
    :param omg_k: (3x12) Y_i set
    :return Z_i: (6x12)
    """

    # step6: Y_i set to Z_i set (6x2n) measurement model
    Z_i = np.zeros((6, 12))

    g = np.asarray([[0], [0], [0], [1]])
    q_inv = quatInv(quat_k)
    z_acc = quatMulti(quat_k, quatMulti(g, q_inv))

    Z_i[:3, :] = quat2vec(z_acc)
    Z_i[3:, :] = omg_k

    return Z_i

##### get the update values and Kalman Gain #####
def update(vals_imu, W_i_prime, Z_i, R):
    """
    :param vals_imu: (7x1) imu data
    :param W_i_prime: (6x12)
    :param Z_i: (6x12)
    :param R: (6x6) measurement noise
    :return:
    """

    # step7: measurement estimate z_k_bar, innnovation v_k, measurement noise R
    z_k_bar = getMean(Z_i)

    P_zz = getCov(Z_i - z_k_bar)
    # difference between measure and actual
    v_k = vals_imu - z_k_bar
    P_vv = P_zz + R

    # step8: cross correlation matrix P_xz, Kalman gain K_k
    P_xz = np.dot(W_i_prime, np.transpose(Z_i - z_k_bar)) / 12.

    K_k = np.dot(P_xz, np.linalg.inv(P_vv))

    return K_k, v_k, P_vv

##### get posterior estimate #####
def correction(P_k_bar, x_hat_k_bar, K_k, v_k, P_vv):
    """
    :param P_k_bar: (6x6) prior covariance
    :param x_hat_k_bar: (7x1) prior estimate
    :param K_k: (6x6) Kalman Gain
    :param v_k: (6x1)
    :param P_vv: (6x6)
    :return:
    """

    # step9: update posterior estimate x_hat_k and estimate error covariance P_k
    P_k = P_k_bar - np.dot(K_k, np.dot(P_vv, np.transpose(K_k)))

    Kv = np.dot(K_k, v_k) # (6x1)
    x_hat_k_ = Kv[3:,[0]] # (3x1)
    Kv_ = vec2quat(Kv[:3,[0]]) # (4x1)

    x_hat_k_prime = x_hat_k_bar[4:,[0]] + x_hat_k_
    Kv_prime = quatMulti(x_hat_k_bar[:4,[0]], Kv_)

    x_hat_k = np.vstack((Kv_prime, x_hat_k_prime))

    return P_k, x_hat_k

##### run UKF model #####
def ufk(P_k, x_hat_k, vals_imu, ts_imu, Q, R):

    # X_i step -> sigma points to quaternion sigma points
    quat_k_bar, omg_k_bar = sigmaPoints(P_k, x_hat_k, Q)

    # Y_i step -> process model: predicts the evolution of state vector transformed sigma points
    quat_k, omg_k = processModel(quat_k_bar, omg_k_bar, ts_imu)

    # prediction: no information from measurement yet
    P_k_bar, x_hat_k_bar, W_i_prime = prediction(quat_k, omg_k)

    # Z_i step -> measurement model: relates the measurement value to the state vector
    Z_i = measurementModel(quat_k, omg_k)

    # Kalman Gain and update equations
    K_k, v_k, P_vv = update(vals_imu, W_i_prime, Z_i, R)

    # correction: calculate posterior estimate
    P_k, x_hat_k = correction(P_k_bar, x_hat_k_bar, K_k, v_k, P_vv)

    return P_k, x_hat_k


if __name__ == "__main__":
    print('')