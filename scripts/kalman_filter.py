import numpy as np
import scipy

class KalmanFilter:
    '''
    Linear Velocity model to track bounding boxes

    state vector : [xcen, ycen, ar, h, vel_xcen, vel_ycen, vel_ar, vel_h]
    where (xcen, ycen) is the center of bounding box, ar is the aspect ratio,
    h is the height of bbox.

    measurement : [xcen, ycen, ar, h]

    '''
    def __init__(self):
        '''
        _std_weight_position : Standard deviation of position values
        _std_weight_velocity : Standard deviation of velocity values
        A : State transition matrix from measuement space to state vectors
        H : transition matrix from state vectors to measurement space
        '''
        self._std_weight_position = 1.0/20
        self._std_weight_velocity = 1.0/160
        self.dt=1                           

        self.A = np.array([[1,0,0,0,self.dt,0,0,0],
                           [0,1,0,0,0,self.dt,0,0],
                           [0,0,1,0,0,0,self.dt,0],
                           [0,0,0,1,0,0,0,self.dt],
                           [0,0,0,0,1,0,0,0],
                           [0,0,0,0,0,1,0,0],
                           [0,0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0,1]])
        
        self.H = np.eye(4,8)

    def initialize(self, measurement):
        '''
        We introduce more variance if the object is closer (indicated with a larger bbox height)

        Variance of ar is smaller and constant (as it doesnt change with constant height)
        '''
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3], #more variance if the the object is closer (more height)
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        
        covariance = np.diag(np.square(std))
        return mean, covariance
        
    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self.A, mean)
        covariance = np.linalg.multi_dot((self.A, covariance, self.A.T)) + motion_cov

        return mean, covariance
    
    def project(self, mean, covariance):
        # Project back to measurement space

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self.H, mean)
        covariance = np.linalg.multi_dot((self.H, covariance, self.H.T)) + innovation_cov

        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        project_mean, project_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            project_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self.H.T).T,
            check_finite=False).T
        innovation = measurement - project_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, project_cov, kalman_gain.T))
        
        return new_mean, new_covariance
    
    def mahalanobis_dist(self, mean, covariance, measurement):
        proj_mean, proj_cov = self.project(mean, covariance)

        cholesky_factor = np.linalg.cholesky(proj_cov)
        d = measurement - proj_mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha