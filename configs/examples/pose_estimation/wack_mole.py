# Copyright (c) OpenMMLab. All rights reserved.
executor_cfg = dict(
    # Basic configurations of the executor
    name='Pose Estimation',
    camera_id=0,
    camera_frame_shape=(960, 720),
    synchronous=False,
    # Define nodes.
    # The configuration of a node usually includes:
    #   1. 'type': Node class name
    #   2. 'name': Node name
    #   3. I/O buffers (e.g. 'input_buffer', 'output_buffer'): specify the
    #       input and output buffer names. This may depend on the node class.
    #   4. 'enable_key': assign a hot-key to toggle enable/disable this node.
    #       This may depend on the node class.
    #   5. Other class-specific arguments
    nodes=[
        # 'PoseTrackerNode':
        # This node performs object detection and pose tracking. Object
        # detection is performed every several frames. Pose estimation
        # is performed for every frame to get the keypoint as well as the
        # interval bbox when object detection is not performed.
        dict(
            type='PoseTrackerNode',
            name='pose tracker',
            det_model_config='model_configs/mmdet/'
            'ssdlite_mobilenetv2_scratch_600e_coco.py',
            det_model_checkpoint='https://download.openmmlab.com'
            '/mmdetection/v2.0/ssd/'
            'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
            'scratch_600e_coco_20210629_110627-974d9307.pth',
            pose_model_config='model_configs/mmpose/'
            'vipnas_mbv3_coco_wholebody_256x192_dark.py',
            pose_model_checkpoint='https://download.openmmlab.com/mmpose/'
            'top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
            '-e2158108_20211205.pth',
            det_interval=10,
            labels=['person'],
            smooth=True,
            device='cuda:0',
            input_buffer='_input_',  # `_input_` is an executor-reserved buffer
            output_buffer='human_pose'),
        # 'ObjectAssignerNode':
        # This node binds the latest model inference result with the current
        # frame. (This means the frame image and inference result may be
        # asynchronous).
        dict(
            type='ObjectAssignerNode',
            name='object assigner',
            frame_buffer='_frame_',  # `_frame_` is an executor-reserved buffer
            object_buffer='human_pose',
            output_buffer='frame'),
        # 'ObjectVisualizerNode':
        # This node draw the pose visualization result in the frame image.
        # Pose results is needed.
        dict(type='ObjectVisualizerNode',
             name='object visualizer',
             enable_key='v',
             enable=False,
             input_buffer='frame',
             output_buffer='mole'),
        dict(
            type='MoleNode',
            name='mole',
            input_buffer='mole',
            output_buffer='vis',
        ), 
        # 'NoticeBoardNode':
        # This node show a notice board with given content, e.g. help
        # information.
        dict(
            type='NoticeBoardNode',
            name='instruction',
            enable_key='h',
            enable=False,
            input_buffer='vis',
            output_buffer='display',
            content_lines=[
                'This is a demo for pose visualization and simple wack-mole '
                'game. Have fun!', '', 'Hot-keys:',
                '"v": Pose estimation result visualization',
                '"h": Show help information',
                '"q": Exit'
            ],
        ),
        # 'RecorderNode':
        # This node save the output video into a file.
        dict(type='RecorderNode',
             name='recorder',
             out_video_file='webcam_demo.mp4',
             input_buffer='display',
             output_buffer='_display_'
             # `_display_` is an executor-reserved buffer
             )
    ])
