from load_dat import *

# determine the bias of the IMU
def bias(imu_dat):

    vals, _ = load_imu(imu_dat)
    # observe from the vicon that it doesn't move in the beginning
    acc = vals[:3,:500]
    omg = vals[3:,:500]

    acc_bias = acc.sum(axis=1)/500.
    omg_bias = omg.sum(axis=1)/500.

    acc_bias[2] = (acc_bias[0]+acc_bias[1])/2.

    acc_bias = acc_bias.reshape((3,1))
    omg_bias = omg_bias.reshape((3,1))

    return acc_bias, omg_bias


def scale(imu_dat):

    # acceleration
    vals, _ = load_imu(imu_dat)

    acc = vals[:3, :]
    acc_bias, _ = bias(imu_dat)

    # calibration
    acc = (acc - acc_bias)**2
    scale_acc = (np.sum(np.sqrt(acc.sum(axis=0))))/np.sum(acc.sum(axis=0))

    # angular velocity
    sensitivity = 3.3

    scale_omg = 3300./1023./(180./np.pi)/sensitivity

    return scale_acc, scale_omg

if __name__ == "__main__":
    acc_bias, omg_bias = bias(1)
    print(acc_bias)
    print(omg_bias)
    scale_acc, scale_omg = scale(1)
    print(scale_acc)
    print(scale_omg)

