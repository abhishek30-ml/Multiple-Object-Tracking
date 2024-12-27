import cv2
import numpy as np
import random

def draw_bbox(frame, det, frame_id, color_map):
    '''
    Draw bbox for given detection file
    '''
    mask = det[:,0]==frame_id
    gt_frame = det[mask]
    for row in gt_frame:
        track_id = int(row[1])
        top_left_x = int(row[2])
        top_left_y = int(row[3])
        bottom_right_x = int(top_left_x + row[4])
        bottom_right_y = int(top_left_y + row[5])

        if track_id not in color_map:
            color_map[track_id] = tuple(random.randint(0, 255) for _ in range(3))
    
        color = color_map[track_id]

        thickness = 2        # Line thickness
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

    return frame, color_map


def det_file(offline_all_tracks, all_tracks, path):
    '''
    Create detection file output
    '''
    det_out = []
    my_tracks = offline_all_tracks+all_tracks
    for track in my_tracks:
        uni_id = track.id
        if len(track.frame)>10:
            start = track.frame[0]
            end = track.frame[-1]
            c=0
            for k in range(start, end+1):
                if c < len(track.mean):
                    frame = k
                    measure = track.mean[c]
                    h = measure[3]
                    w = measure[2]*h
                    x1 = int(measure[0]-w/2)
                    y1 = int(measure[1]-h/2)
                    det_out.append([frame, uni_id, x1, y1, w, h, -1, -1, -1, -1])
                    c=c+1

    det_out = np.array(det_out)
    save_path = path + "/det.txt"

    # Save the array to a text file
    np.savetxt(save_path, det_out, fmt='%d', delimiter=',')

    return det_out