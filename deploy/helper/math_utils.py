import math
import numpy as np

class MathFunc:
    @staticmethod
    def m_to_mm(length):
        """
        length: [m]
        """
        return length * 1000.
    
    @staticmethod
    def mm_to_m(length):
        """
        length: [mm]
        """
        return length / 1000.

    @staticmethod
    def degree_to_rad(angle):
        """
        angle: [degree]
        """
        return angle * np.pi / 180.
    
    @staticmethod
    def rad_to_degree(angle):
        """
        angle: [rad]
        """
        return angle * 180. / np.pi
    
    @staticmethod
    def single_axis_rotMat(axis, angle):
        """
        axis: x / y / z
        angle: [rad]
        """
        assert axis in ['x', 'y', 'z'], "Unavailable axis"
        
        c = np.cos(angle)
        s = np.sin(angle)

        if axis == 'x':
            return np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]], dtype=np.float32)
        elif axis == 'y':
            return np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]], dtype=np.float32)
        else:
            return np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]], dtype=np.float32)

    @staticmethod
    def euler_to_rotMat(euler_x, euler_y, euler_z):
        """
        euler_x, euler_y, euler_z: [rad]
        """
        R_x = MathFunc.single_axis_rotMat('x', euler_x)
        R_y = MathFunc.single_axis_rotMat('y', euler_y)
        R_z = MathFunc.single_axis_rotMat('z', euler_z)
        return R_z @ R_y @ R_x
    
    @staticmethod
    def rotMat_to_euler(rotMat):
        assert(MathFunc.is_rotMat(rotMat)), "Given matrix is not rotation matrix."
        sy = math.sqrt(rotMat[0,0] * rotMat[0,0] + rotMat[1,0] * rotMat[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(rotMat[2,1] , rotMat[2,2])
            y = math.atan2(- rotMat[2,0], sy)
            z = math.atan2(rotMat[1,0], rotMat[0,0])
        else :
            x = math.atan2(- rotMat[1,2], rotMat[1,1])
            y = math.atan2(- rotMat[2,0], sy)
            z = 0
        return np.array([x, y, z])
    
    @staticmethod
    def rotMat_to_quat(R):
        """
        Convert a rotation matrix to a quaternion.

        Parameters:
        R (numpy.ndarray): 3x3 rotation matrix.

        Returns:
        numpy.ndarray: Quaternion as [w, x, y, z].
        """
        # Ensure the matrix is of the correct shape
        assert R.shape == (3, 3), "Rotation matrix must be 3x3"

        # Calculate the trace of the matrix
        trace = np.trace(R)

        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        quaternion = np.array([w, x, y, z])
        return quaternion

    
    @staticmethod
    def is_rotMat(rotMat):
        Rt = np.transpose(rotMat)
        shouldBeIdentity = np.dot(Rt, rotMat)
        I = np.identity(3, dtype = rotMat.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-4
    
    @staticmethod
    def euler_xyz_to_quat(roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        # compute quaternion
        qw = cy * cr * cp + sy * sr * sp
        qx = cy * sr * cp - sy * cr * sp
        qy = cy * cr * sp + sy * sr * cp
        qz = sy * cr * cp - cy * sr * sp
        return np.array([qw, qx, qy, qz])
    
    @staticmethod
    def quat_to_euler_xyz(quat):
        q_w, q_x, q_y, q_z = quat[0], quat[1], quat[2], quat[3]
        # roll (x-axis rotation)
        sin_roll = 2.0 * (q_w * q_x + q_y * q_z)
        cos_roll = 1 - 2 * (q_x * q_x + q_y * q_y)
        roll = np.arctan2(sin_roll, cos_roll)

        # pitch (y-axis rotation)
        sin_pitch = 2.0 * (q_w * q_y - q_z * q_x)
        pitch = np.where(np.abs(sin_pitch) >= 1, np.pi / 2. * np.sign(sin_pitch), np.arcsin(sin_pitch))

        # yaw (z-axis rotation)
        sin_yaw = 2.0 * (q_w * q_z + q_x * q_y)
        cos_yaw = 1 - 2 * (q_y * q_y + q_z * q_z)
        yaw = np.arctan2(sin_yaw, cos_yaw)

        return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
    
    @staticmethod
    def quat_unique(q):
        return np.where(q[0] < 0, -q, q)
    
    @staticmethod
    def quat_conjugate(q):
        return np.concatenate((q[0:1], -q[1:]))
    
    @staticmethod
    def quat_mul(q1, q2):
        # reshape to (N, 4) for multiplication
        # shape = q1.shape
        # extract components from quaternions
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        # perform multiplication
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        return np.array([w, x, y, z])
    
    @staticmethod
    def quat_to_axis_angle(quat, eps: float = 1.0e-6):
        quat = quat * (1.0 - 2.0 * (quat[0:1] < 0.0))
        mag = np.linalg.norm(quat[1:])
        half_angle = np.arctan2(mag, quat[0])
        angle = 2.0 * half_angle
        # check whether to apply Taylor approximation
        sin_half_angles_over_angles = np.where(
            np.abs(angle) > eps, np.sin(half_angle) / angle, 0.5 - angle * angle / 48
        )
        return quat[1:4] / sin_half_angles_over_angles
    
    @staticmethod
    def quat_error_magnitude(q1, q2):
        quat_diff = MathFunc.quat_mul(q1, MathFunc.quat_conjugate(q2))
        return np.linalg.norm(MathFunc.quat_to_axis_angle(quat_diff))

