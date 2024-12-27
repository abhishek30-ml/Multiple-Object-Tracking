import numpy as np
from scripts.track import Track

def new_track(unmatches, des_list, uniq_id, frame_no, kf):
    '''
    Create track object for the unmatched tracks
    '''
    unmatches_track = []
    for j,measur in enumerate(unmatches):
        mean, covariance = kf.initialize(measur)
        des1 = des_list[j]
        if des_list[j] is None or des_list[j].shape[0]<11:
            des1 = None
        unmatches_track.append(Track(uniq_id,'new',[measur], [frame_no] ,[mean], covariance, [des1]))
        uniq_id+=1

    return unmatches_track, uniq_id


def collect_measurement(results):
    '''
    change bbox from detection model to measurement format
    [xcen, ycen, ar, h]
    '''
    mesur_list = []
    for obj in results[0].boxes:

        bbox = obj.xywh[0]          
        x_cen = float(bbox[0])
        y_cen = float(bbox[1])
        ar = float(bbox[2]/bbox[3])
        h = float(bbox[3])
        measurement = np.array([x_cen, y_cen, ar, h])
        mesur_list.append(measurement)

    return mesur_list

