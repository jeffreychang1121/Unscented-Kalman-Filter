from scipy.io import loadmat
import numpy as np

def load_imu(imu_dat):

    data_folder = str(imu_dat)
    file_to_open = './imu/imuRaw' + data_folder + '.mat'
    imu = loadmat(file_to_open)

    vals = np.asarray(imu['vals'])
    ts = np.asarray(imu['ts'])
    return vals, ts

def load_vicon(vicon_dat):

    data_folder = str(vicon_dat)
    file_to_open = './vicon/viconRot' + data_folder + '.mat'
    vicon = loadmat(file_to_open)

    rots = np.asarray(vicon['rots'])
    ts = np.asarray(vicon['ts'])
    return rots, ts

if __name__ == "__main__":
    print('')