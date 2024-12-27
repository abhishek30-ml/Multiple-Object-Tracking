import numpy as np
from scipy.optimize import linear_sum_assignment

def maha_dist_matrix(mesur_list, all_tracks, kf):
    '''
    Calculate cost matrix C and gate matrix B btw measurement 
    and kalman pred from t-1 for mahalanobis dist
    '''
    C = np.zeros((len(mesur_list), len(all_tracks)))
    B = np.zeros((len(mesur_list), len(all_tracks)))
    for i in range(len(mesur_list)):
        for j in range(len(all_tracks)):
            C[i][j] = kf.mahalanobis_dist(all_tracks[j].mean[-1], all_tracks[j].covariance, mesur_list[i])
            if C[i][j] <= 9.4877:
                B[i][j] = 1
            else:
                B[i][j] = 0
    return C,B


def sift_dist_matrix(des_list, all_tracks, sift, min_score, compare_number):
    '''
    Calculate cost matric C and gate matrix B btw measurement
    and kalman pred from t-1 for sift scores

    Reject descriptors that are less than 10 in number

    For a given measurement, take the maximum score when
    compared with previous 3 time steps descriptors
    '''
    C = np.zeros((len(des_list), len(all_tracks)))
    B = np.zeros((len(des_list), len(all_tracks)))
    
    for i in range(len(des_list)):
        if des_list[i] is None or des_list[i].shape[0]< 11:
            continue
        for j in range(len(all_tracks)):
            max_val=0
            for k in all_tracks[j].descriptor[-compare_number:]:
                if k is None:
                    continue
                val = sift.percent_matching(des_list[i], k)
                if val>max_val:
                    max_val = val
            C[i,j] = max_val
            if C[i,j] >= min_score:
                B[i,j] = 1

    return C,B

def matching_assignment(C, B, C2, B2, all_tracks, unmatches, des_list, frame_no, kf):
    '''
    C,B : 'mahalanobis dist' cost and gate matrices
    C2,B2 : 'sift score' cost and gate matrices

    Performs Linear sum assignment between new measurments(unmatches)
    and all current tracks.

    Remove the matched tracks from unmatches and descriptor list
    and append the matches to existing tracks

    Use Kalman filter correction to update the mean and covariance
    of matched tracks

    Returns the updated unmatches, all_tracks and descriptor list.
    '''
    l1 = 0.2
    l2 = 0.8
    C2 = 1 - C2/100
    cost = l1*C + l2*C2
    row_ind, col_ind = linear_sum_assignment(cost)
    del_ind = []
    for k in range(len(row_ind)):
        if B[row_ind[k]][col_ind[k]]>0 and B2[row_ind[k]][col_ind[k]]>0:
            obj = all_tracks[col_ind[k]]
            obj.measurement.append(unmatches[row_ind[k]])  #append the measurement that matches.
            obj.descriptor.append(des_list[row_ind[k]])    #append the descriptor that matches
            obj.frame.append(frame_no) 
            obj.status = 'matched'
            obj.reset()

                # Use Kalman to update the values for Matched
            new_m, new_c = kf.update(obj.mean[-1], obj.covariance, obj.measurement[-1])
            obj.mean[-1] = new_m
            obj.covariance = new_c

            del_ind.append(row_ind[k])

    unmatches = [ele for idx, ele in enumerate(unmatches) if idx not in del_ind]
    des_list = [ele for idx, ele in enumerate(des_list) if idx not in del_ind]

    return all_tracks, unmatches, des_list
                