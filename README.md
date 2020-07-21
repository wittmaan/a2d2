# a2d2

## Setup

- conda create --name <env> --file requirements.txt
- download dataset https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic_bboxes.tar
- extract it to the data folder

## Approach

Use the approach form this paper 3D Bounding Box Estimation Using Deep Learning and Geometry (https://arxiv.org/pdf/1612.00496.pdf)

Regress on stable 3d object properties using deep convolutional neural netword and then 
combine these estimates with geometric constrains provided by a 2D object bounding box
to produce a complete 3D bounding box.

first network output estimates 3D object orientation
second network output estimates 3D object dimensions

method takes the 2D detection bounding box and estimates a 3D bounding box

## Problem

Given the LIDAR and CAMERA data, determine the location and orientation in 3D of other vehicles around the car.

## Solution

2D object detection on camera image is easy and can be solved by various CNN-based solutions like YOLO and RCNN. The tricky part here is the 3D requirement. It becomes a little easier with LIDAR data, accompanied with the calibration matrices.

So the solution is straight-forward in three processing steps:

- Detect 2D BBoxes of other vehicles visible on image frame captured by CAMERA. This can be achieved by YOLOv2 or SqueezeDet. It turns out that SqueezeDet works better for this job and is selected.
- Determine the dimension and the orientation of detected vehicles. As demonstrated by [https://arxiv.org/abs/1612.00496](https://arxiv.org/abs/1612.00496), dimension and orientation of other vehicles can be regressed from the image patch of corresponding 2D BBoxes.
- Determine the location in 3D of detected vehicles. This can be achived by localizing the point cloud region whose projection stays within the detected 2D BBoxes.


## References

- https://github.com/lzccccc/3d-bounding-box-estimation-for-autonomous-driving
- https://github.com/smallcorgi/3D-Deepbox