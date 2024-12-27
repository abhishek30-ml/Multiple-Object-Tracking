import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

import argparse
import os

from scripts import initialize
from scripts import matching
from scripts import tracker
from scripts import result
from scripts.sift_descriptor import Sift 
from scripts.kalman_filter import KalmanFilter


def run(data_path, model, detection_conf, sift_good_dist, min_sift_score, accumulate_sift, visualize):

    image_folder = data_path + '/img1/'
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

    model = YOLO("models/" + model +".pt")
    kf = KalmanFilter()
    sift = Sift(sift_good_dist)

    offline_all_tracks = []
    all_tracks=[]
    uniq_id = 1

    for frame_no, image in enumerate(images):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        results = model(frame, verbose=False, conf=detection_conf)

        # Collect measurements
        mesur_list = initialize.collect_measurement(results)

        # Collect feature descriptors
        des_list = sift.collect_descriptors(mesur_list, frame)

        # Cal cost matrix 'C' btw measurement and kalman pred from t-1
        if all_tracks:
            C,B = matching.maha_dist_matrix(mesur_list, all_tracks, kf)

        # Cal value of SIFT descriptors btw measurements and kalman pred from t-1
        if all_tracks:
            C2, B2 = matching.sift_dist_matrix(des_list, all_tracks, sift, min_sift_score, accumulate_sift)

        # Perform Matching Cascade btw new mesur and previous tracks - Update lists - matches and unmatches
        unmatches = mesur_list.copy()
        if all_tracks:
            all_tracks, unmatches, des_list = matching.matching_assignment(C, B, C2, B2, all_tracks, unmatches, des_list, frame_no, kf)

        # Create new tracks for Unmatched.
        unmatches_track, uniq_id = initialize.new_track(unmatches, des_list, uniq_id, frame_no, kf)

        # Use kalman filter to predict 
        all_tracks, offline_all_tracks = tracker.update_track(all_tracks, unmatches_track, offline_all_tracks, kf)

    det_array = result.det_file(offline_all_tracks, all_tracks, data_path)

    if visualize:
        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)

        height, width, layers = frame.shape
        frame_size = (width, height)

        video_out_path = video_out_path = data_path + '/output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        out = cv2.VideoWriter(video_out_path, fourcc, 30.0, frame_size)  # 30 FPS
        color_map={}
        # Loop through all images and write them to the video
        for i,image in enumerate(images):
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            frame,color_map = result.draw_bbox(frame, det_array, i,color_map)
            out.write(frame)

        # Release the video writer object
        out.release()





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=None, required=True, type=str)
    parser.add_argument('--model', default='yolo11n', type=str)
    parser.add_argument('--detection_conf', default=0.4, type=float)
    parser.add_argument('--sift_good_dist', default=300.0, type=float)
    parser.add_argument('--min_sift_score', default=20.0, type=float)
    parser.add_argument('--accumulate_sift', default=3, type=int)
    parser.add_argument('--visualize', default=True, type=bool)

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    run(args.data_path, args.model, args.detection_conf, 
        args.sift_good_dist, args.min_sift_score, 
        args.accumulate_sift, args.visualize)

