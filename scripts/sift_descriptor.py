import numpy as np
import cv2

class Sift:
    '''
    Simple sift operator to generate descriptor vector of the bbox object

    Can be replaced with surf for faster performance
    '''
    def __init__(self, threshold):
        self.threshold = threshold

    def collect_descriptors(self, mesur_list, img):
        '''
        Outputs List of descriptors for the corresponding detected objects
        '''
        des_list = []
        sift = cv2.SIFT_create()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in mesur_list:
            h = i[3]
            w = i[2]*h
            x1 = int(i[0]-w/2)
            y1 = int(i[1]-h/2)
            x2 = int(i[0] + w/2)
            y2 = int(i[1] + h/2)
            template = img[y1:y2, x1:x2]
            template = cv2.GaussianBlur(template, (5, 5), 0)
            _, des = sift.detectAndCompute(template,None)
            des_list.append(des)

        return des_list
    
    def percent_matching(self, des1, des2):
        '''
        Descriptor matching with score between 0 to 100 as output. 
        Higher the score, better the match.

        Perform ratio test and reject matches with ratio of distance
        between first and second closest match greater than 0.95 

        Calculate distance between matches and count number of matches
        with distance less than 300

        Output ratio of count of thresholded distance and total matches
        '''
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good_dist = []
        i=0
        for m,n in matches:
            if m.distance < 0.95*n.distance:
                good_dist.append(m.distance)
            i=i+1
        
        good_dist = np.array(good_dist)
        thres_dist = good_dist[good_dist < self.threshold]
        if len(good_dist)==0:
            return 0
        
        return (len(thres_dist)/len(good_dist))*100
    