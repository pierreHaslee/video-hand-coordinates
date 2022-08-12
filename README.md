# video-hand-coordinates

## This github contains two main scripts:
- video_to_hands.py:
  > [outputs hand coordinates from videos](#getting-hand-coordinates-from-videos)
- show_hands_coords.py:
  > [shows video with coordinates and arrows plotted for the speed and acceleration of each keypoint](#show-coordinates-with-video)
  
/!\ check the [requirements](#requirements)

## Getting hand coordinates from videos

### Usage

```
python video_to_hands.py\
<path to mmpose git folder>/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth\
<path to mmpose git folder>/configs/hand/2d_kpt_sview_rgb_img/deeppose/onehand10k/res50_onehand10k_256x256.py deeppose_res50_onehand10k_256x256-cbddf43a_20210330.pth\
--video-root <folder path of input videos> --out-video-root [folder path of outputed annotated videos]\
[--device <cpu/gpu>] [--show] --savepath <folder path of the outputed coordinates>
```

### Input format

`--video-root` argument, give the folder in which are your `.mp4 / .MP4` videos

### Output format

`--savepath` argument, give the folder in which to save coordinates files.

This python script outputs a pickle file containing all relevant hand coordinates.
This is its structure:

    The pickle file is a list, each element of the list correspond to one frame of the video.
  
    For each frame, the list element is a python dictionnary containing 0, 1 or 2 keys : none, left and/or right.
    The dictionnary values is another dictionnary with 2 keys: 'keypoints' and 'bbox'.
    The values for 'keypoints' is a list of the hand keypoints, and the value for 'bbox' contains the detection bounding box for the hand and its inference confidence.
  
  *i.e.*
  `pickle_list[2649]['left'][12] is the 12th keypoint of the left hand visible on the 2649th frame`

## Showing coordinates on videos

### Usage

```
python show_hands_coords.py <path of the video> <path of the coordinates file> [--out-video <path of the outputed video file>]
```

## Requirements 

In order to use this github, you need to clone and install the following github repositories in a conda environment:

You need to clone the [mmpose github repo](https://github.com/open-mmlab/mmpose) and the [mmdetection github repo](https://github.com/open-mmlab/mmdetection) and perform the installation process on each of them:
- [install instructions for mmpose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md)
- [install instructions for mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)

**If you work on linux:**
All of the installation steps above (including git clonings) can be performed by the `install_for_you.sh` bash script provided in this github.
> Be sure that you have *conda* installed, that you don't already have a *conda environment* named 'openmmlab', and that you run this script at the root of this github repository.

You will also need to download two weight files to be loaded in the machine learning models (you can put them in a 'weights' folder):
  - [deeppose_res50_onehand10k_256x256-cbddf43a_20210330.pth](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_onehand10k_256x256-cbddf43a_20210330.pth) for the hand pose model
  - [cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth) for the hand detection model

**optional:** create these folders *weights* *video* *output*, to store respectively the weights downloaded above, your videos, and your outputted pickle files.
