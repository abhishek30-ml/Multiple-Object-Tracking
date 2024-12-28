# Multiple Object Tracking with Yolo, SIFT and Kalman Filter

https://github.com/user-attachments/assets/9125518d-e6be-4da6-902b-681c2ab1f78b

## TDS
[Link here!](https://medium.com/@abhishek.sabnis2000/multiple-object-tracking-with-yolo-sift-and-kalman-filter-684088268e8e) for a detailed explaination of the theory.


## Introduction
This repository contains code to implement online detection-based Multiple Object Tracking.
We use Yolov11 to detect bounding boxes. Combination of Kalman Filter with SIFT descriptor is implemented to capture the motion model and appearance model.

## Dependencies

- numpy
- OpenCV
- Yolo

## Installation

1. Clone the repository: **`git clone https://github.com/abhishek30-ml/Multiple-Object-Tracking.git`**
2. Navigate to the project directory: **`cd Multiple-Object-Tracking`**
3. Place your data in image format in directory: **`data/my_video/img1/ `**
4. Download the yolo model weights and place them in directory **`models/yolov11n.pt `**

## Running the code

**`python3 mot.py 
  --data_path = data/my_video 
  --model = yolo11n 
  --detection_conf = 0.4 
  --sift_good_dist = 300 
  --min_sift_score = 20 
  --accumulate_sift = 3
  --visualize = True`**

Check the detection file output at **`data/my_video/det.txt `** and video results at **`data/my_video/output.mp4 `**

## Overview of script files

The main script to execute is **`mot.py`**. Supporting scripts are 
* **`initialize.py`** : Creates new tracks and records new measurment
* **`kalman_filter.py`** : Kalman Filter implementaion of linear motion model and calculation of mahalanobis dist
* **`matching.py`** : Calculates cost matrices and performs linear matching
* **`result.py`** : Creates output detection file (det.txt) and draws bbox
* **`sift_descriptor.py`** : Computes SIFT descriptor and performs score calculation of matching score
* **`track.py`** : Track class that stores information about each track
* **`tracker.py`** : Performs kalman prediction of all tracks and updates the track status

## Example 2

https://github.com/user-attachments/assets/c7af048c-808b-4625-a2b3-bed0667e5dad

Furthur improvements required when the motion is fast and number of occlusions increase. Possible implementations include non-linear kalman filter along with a better appearance descriptor.

