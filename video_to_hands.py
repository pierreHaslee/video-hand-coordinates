# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import pickle
from data_utils import combine_results, coords_to_hands

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-root', type=str, help='Video folder')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--savepath',
        default='list_results.pickle',
        help='the name of the outputed resulting coordinates'
    )

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    print('listing all mp4 files that will be analyzed')
    files = os.listdir(args.video_root)
    files = list(map(lambda x: os.path.join(args.video_root,x),files))
    videos = list(filter(lambda x: x.split('.')[-1] == 'mp4',files))
    for video_path in videos:
        print(video_path)

    for video_path in videos:

        # read video
        video = mmcv.VideoReader(video_path)
        assert video.opened, f'Failed to load video file {video_path}'
        

        if args.out_video_root == '':
            save_out_video = False
        else:
            os.makedirs(args.out_video_root, exist_ok=True)
            save_out_video = True

        if save_out_video:
            fps = video.fps
            size = (video.width, video.height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            videoWriter = cv2.VideoWriter(
                os.path.join(args.out_video_root,
                            f'vis_{os.path.basename(video_path)}'), fourcc,
                fps, size)

        frame_x_width = video.width

        # frame index offsets for inference, used in multi-frame inference setting
        if args.use_multi_frames:
            assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
            indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

        # whether to return heatmap, optional
        return_heatmap = False

        # return the output of some desired layers,
        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        result_list = list()

        print('Running inference...')
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            # get the detection results of current frame
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, cur_frame)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            for person in person_results:
                print(person['bbox'])

            if args.use_multi_frames:
                frames = collect_multi_frames(video, frame_id, indices,
                                            args.online)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                frames if args.use_multi_frames else cur_frame,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            result_list.append(combine_results(person_results, pose_results))

            # show the results
            vis_frame = vis_pose_result(
                pose_model,
                cur_frame,
                pose_results,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=False)

            if args.show:
                cv2.imshow('Frame', vis_frame)

            if save_out_video:
                videoWriter.write(vis_frame)

            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if save_out_video:
            videoWriter.release()
        if args.show:
            cv2.destroyAllWindows()

        dirpath = os.path.dirname(os.path.abspath(__file__))
        savepath = os.path.join(dirpath, args.savepath, os.path.split(video_path)[1][:-4]+'_hands_coords.pickle')

        coords_hands_video = coords_to_hands(result_list, frame_x_width)

        with open(savepath, 'wb') as f:
            pickle.dump(coords_hands_video,f)


if __name__ == '__main__':
    main()
