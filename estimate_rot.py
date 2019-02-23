#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter

from load_dat import *
from calibrate import *
from ukf_model import *
import numpy as np
import matplotlib.pyplot as plt

def estimate_rot(data_num=1):

	# load IMU, VICON data
	vals, ts = load_imu(data_num)
	acc_bias, omg_bias = bias(data_num)
	acc_scale, omg_scale = scale(data_num)


	# calibrate IMU data
	imu_acc = (vals[:3,:] - acc_bias) * acc_scale
	imu_omg = (vals[3:,:] - omg_bias) * omg_scale

	imu = np.zeros((6,len(ts[0])))
	imu[0,:] = -imu_acc[0,:]
	imu[1,:] = -imu_acc[1,:]
	imu[2,:] = imu_acc[2,:]
	imu[3,:] = imu_omg[1,:]
	imu[4,:] = imu_omg[2,:]
	imu[5:,:] = imu_omg[0,:]

	# initialization
	P_k = np.eye(6)
	x_hat_k = np.asarray([1, 0, 0, 0, 0, 0, 0]).reshape((-1, 1))

	process_noise = 5e-7
	acc_noise = 9e-3
	gyro_noise = 10e-9

	Q = np.eye(6)
	Q[:3,:3] += np.ones(3)
	Q[3:,3:] += np.ones(3)
	Q = Q*process_noise
	R = np.eye(6)
	R[:3, :3] += np.ones(3)
	R[3:, 3:] += np.ones(3)
	R[:3,:3] *= acc_noise
	R[3:,3:] *= gyro_noise

	# store quaternion for each time
	quat = np.zeros((4,len(ts[0])))

	for i in range(len(ts[0])):
		# imu data and duration
		vals_imu = imu[:,i].reshape((-1,1))
		if i-1 < 0:
			ts_imu = ts[0,1]-ts[0,0]
		else:
			ts_imu = ts[0,i]-ts[0,i-1]

		P_k, x_hat_k = ufk(P_k, x_hat_k, vals_imu, ts_imu, Q, R)
		# print(x_hat_k)

		quat[:,[i]] = x_hat_k[:4,[0]]

	# compare with Vicon
	rpy_imu = quat2rpy(quat)

	rots, ts_vicon = load_vicon(data_num)
	rpy_vicon = rot2rpy(rots)

	ax = plt.subplot(3,1,1)
	ax.plot(ts_vicon[0],rpy_vicon[0])
	ax.plot(ts[0],rpy_imu[0])
	ax = plt.subplot(3,1,2)
	ax.plot(ts_vicon[0],rpy_vicon[1])
	ax.plot(ts[0],rpy_imu[1])
	ax = plt.subplot(3,1,3)
	ax.plot(ts_vicon[0],rpy_vicon[2])
	ax.plot(ts[0],rpy_imu[2])
	plt.show()

	roll = rpy_imu[0]
	pitch = rpy_imu[1]
	yaw = rpy_imu[2]

	return roll, pitch, yaw


if __name__ == "__main__":
	r, p, y = estimate_rot(data_num=3)
