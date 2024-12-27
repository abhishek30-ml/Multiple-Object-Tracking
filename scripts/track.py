class Track:
    def __init__(self, id, status, measurement, frame, mean, covariance, descriptor):
        '''
        id : int
            Unique id assigned to the track
        status : str
            'new' if new track. 'matched' if track is matching with an older track
        counter : int
            Age of track. (number of frames the track is unmatched)
        measurement: List of ndarray
            List of all observed values from yolo detection
        frame : List of int
            List of frame number the track belongs to
        mean : List of ndarray
            List of all predicted means from Kalman filter
        covariance : ndarray
            predicted covariance matrix from kalman filter
        descriptor : List of ndarray
            List of all descriptors of the tracked object
        
        '''
        self.id = id
        self.status = status # Unmatched, matched
        self.counter=0
        self.measurement = measurement   
        self.frame = frame
        self.mean = mean                
        self.covariance = covariance
        self.descriptor = descriptor    
    
    def reset(self):
        '''
        Reset the age of the track
        '''
        self.counter=0

    def inc_count(self):
        '''
        Increase the age of the track
        '''
        self.counter += 1
