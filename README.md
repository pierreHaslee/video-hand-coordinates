# video-hand-coordinates

## video_to_hands.py

This python script outputs a pickle file storing a list of dict, those dict can contain 2 values : right or left.
The values of those dicts are lists containing the corresponding hand keypoints.

### Usage


```
python video_to_hands.py\
[path to mmpose git folder]/mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth\
[path to mmpose git folder]/mmpose/configs/hand/2d_kpt_sview_rgb_img/deeppose/onehand10k/res50_onehand10k_256x256.py deeppose_res50_onehand10k_256x256-cbddf43a_20210330.pth\
--video-root [folder path of input videos] --out-video-root [folder path of outputed annotated videos]\
--device [cpu/gpu] --show --savepath [folder path of the outputed coordinates]
```