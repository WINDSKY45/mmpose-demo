# Copyright (c) OpenMMLab. All rights reserved.
from itertools import groupby
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from mmpose.apis.webcam.utils.pose import get_hand_keypoint_ids

from mmpose.apis.webcam.utils import get_eye_keypoint_ids, load_image_from_disk_or_url
from mmpose.apis.webcam.nodes.base_visualizer_node import BaseVisualizerNode
from mmpose.apis.webcam.nodes.registry import NODES


@NODES.register_module()
class MoleNode(BaseVisualizerNode):
    """
    Args:
        name (str): The node name (also thread name)
        input_buffer (str): The name of the input buffer
        output_buffer (str|list): The name(s) of the output buffer(s)
        enable_key (str|int, optional): Set a hot-key to toggle enable/disable
            of the node. If an int value is given, it will be treated as an
            ascii code of a key. Please note:
                1. If enable_key is set, the bypass method need to be
                    overridden to define the node behavior when disabled
                2. Some hot-key has been use for particular use. For example:
                    'q', 'Q' and 27 are used for quit
            Default: ``None``
        enable (bool): Default enable/disable status. Default: ``True``.
        kpt_thr (float): The score threshold of valid keypoints. Default: 0.5
        resource_img_path (str, optional): The resource image path or url.
            The image should be a pair of sunglasses with white background.
            If not specified, the url of a default image will be used. See
            ``SunglassesNode.default_resource_img_path``. Default: ``None``

    Example::
        >>> cfg = dict(
        ...    type='SunglassesEffectNode',
        ...    name='sunglasses',
        ...    enable_key='s',
        ...    enable=False,
        ...    input_buffer='vis',
        ...    output_buffer='vis_sunglasses')

        >>> from mmpose.apis.webcam.nodes import NODES
        >>> node = NODES.build(cfg)
    """

    # The image attributes to:
    # "https://www.vecteezy.com/vector-art/1932353-summer-sunglasses-
    # accessory-isolated-icon" by Vecteezy
    # default_resource_img_path = (
    #     'https://user-images.githubusercontent.com/15977946/'
    #     '170850839-acc59e26-c6b3-48c9-a9ec-87556edb99ed.jpg')
    mole_img_path = ('/home/hx/mmpose-demo/mmpose-demo/mole.png')
    hole_img_path = ('/home/hx/mmpose-demo/mmpose-demo/hole.png')

    def __init__(self,
                 name: str,
                 input_buffer: str,
                 output_buffer: Union[str, List[str]],
                 enable_key: Optional[Union[str, int]] = None,
                 enable: bool = True,
                 ):

        super().__init__(
            name=name,
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            enable_key=enable_key, enable=enable)
        self.mole = np.array([0, 0, 0])

        self.resource_img = load_image_from_disk_or_url(self.mole_img_path, cv2.IMREAD_UNCHANGED)
        self.resource_img = cv2.resize(self.resource_img, (140, 140))
        self.hole_img = load_image_from_disk_or_url(self.hole_img_path, cv2.IMREAD_UNCHANGED)
        self.hole_img = cv2.resize(self.hole_img, (140, 140))

        self.mole_mask = [[300,100],[300,300],[300,500]]
        self.evaluate_metric_y = [[120,220], [320,420], [520,620]]
        self.evaluate_metric_x = [330,430]
        self.hits = 0

    def draw(self, input_msg):
        for i in range(3):
            if self.mole[i] > 0:
                self.mole[i] += 1

        if self.mole.sum() == 0 or self.mole[self.mole > 0].min() > 100:
            pos = np.random.choice(range(3))
            self.mole[pos] += 1

        canvas = input_msg.get_image()
        canvas = cv2.flip(canvas, 1)
        h, w = canvas.shape[:2]

        objects = input_msg.get_objects(lambda x: 'keypoints' in x)


        for model_cfg, group in groupby(objects,
                                        lambda x: x['pose_model_cfg']):

            obj = list(group)[0]
            left_hand_kpts = obj['keypoints'][91:112].mean(axis=0)
            right_hand_kpts = obj['keypoints'][112:].mean(axis=0)
            left_hand_kpts[0] = w - left_hand_kpts[0]
            right_hand_kpts[0] = w - right_hand_kpts[0]

            if left_hand_kpts[2] > 0.1:
                self.hits += self.determine_hit(left_hand_kpts)

            if right_hand_kpts[2] > 0.1:
                self.hits += self.determine_hit(right_hand_kpts)

        canvas = self.show_mole(canvas, self.mole)
        cv2.putText(canvas, f'SCORE: {self.hits}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        return canvas

    def determine_hit(self, hand_kpts):
        hit = 0
        for i, m in enumerate(self.mole):
            if m > 0:
                if hand_kpts[0] > self.evaluate_metric_y[i][0] and hand_kpts[0] < self.evaluate_metric_y[i][1] and hand_kpts[1] > self.evaluate_metric_x[0] and hand_kpts[1] < self.evaluate_metric_x[1]:
                    hit += 1
                    self.mole[i] = 0
        return hit

    def show_mole(self, canvas: np.ndarray, mole) -> np.ndarray:
        """

        Args:
            canvas (np.ndarray): The image to apply the effect
            mole

        Returns:
            np.ndarray: Processed image
        """
        patch_size = self.resource_img.shape[0]

        for i, m in enumerate(mole):
            if m > 0:
                x, y = self.mole_mask[i]
                canvas[x:x+patch_size, y:y+patch_size] = (canvas[x:x + patch_size, y:y + patch_size] * (1 - self.resource_img[:, :, 3:] / 255) +
                   self.resource_img[:, :, :3] * (self.resource_img[:, :, 3:] / 255)).astype('uint8')
            else:
                x, y = self.mole_mask[i]
                canvas[x:x+patch_size, y:y+patch_size] = (canvas[x:x + patch_size, y:y + patch_size] * (1 - self.hole_img[:, :, 3:] / 255) +
                   self.hole_img[:, :, :3] * (self.hole_img[:, :, 3:] / 255)).astype('uint8')
        return canvas
