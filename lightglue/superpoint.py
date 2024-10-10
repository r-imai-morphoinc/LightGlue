# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger

import torch
from kornia.color import rgb_to_grayscale
from torch import nn

from .utils import Extractor


def simple_nms(scores, nms_radius: int, add_supp: bool):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    supp_num = 2 if add_supp else 0

    zeros = torch.zeros_like(scores)
    # ---
    max_mask = scores == max_pool(scores)
    # ---
    for _ in range(supp_num):
        # ---
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        # ---
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
        # ---
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        # print("k >= len(keypoints)")
        scores, indices = torch.sort(scores, descending=True)
        return keypoints[indices], scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape

    # keypointsA = keypoints
    # keypointsA = torch.round(keypointsA).to(torch.int32)
    # print(f"keypointsA {keypointsA}")
    # keypointsA = keypointsA // s
    # print(f"keypointsA2 {keypointsA}")
    # for i in range(keypointsA.shape[1]):
    #     pt = keypointsA[0, i, :]
    #     desc = descriptors[0, :, pt[1], pt[0]]
    #     print(desc[:10])

    # print(f"keypoints {keypoints}")
    keypoints = keypoints - s / 2 + 0.5
    # print(f"keypoints2 {keypoints}")
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    # print(f"keypoints3 {keypoints}")
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    # print(f"keypoints4 {keypoints}")

    # print(f"desc(0,0) {descriptors[0, :10, 0, 0]}")
    # print(f"desc(h-1,w-1) {descriptors[0, :10, h-1, w-1]}")
    # print(f"desc(0,w/2) {descriptors[0, :10, 0, w // 2]}")
    # print(f"desc(0,w/2-1) {descriptors[0, :10, 0, w // 2 - 1]}")
    # print(f"desc(0,w-1) {descriptors[0, :10, 0, w-1]}")
    # print(f"desc(h-1,0) {descriptors[0, :10, h-1, 0]}")
    # keypoints[:, :, 0] = 0.0
    # keypoints[:, :, 1] = -1.0

    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    # print(f"descriptors {descriptors}")
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    # print(f"descriptors2 {descriptors}")
    return descriptors


class SuperPoint(Extractor):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """

    default_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": None,
        "detection_threshold": 0.0005,
        "remove_borders": 4,
        "nms_supp": True,
        "antialias": True,
    }

    preprocess_conf = {
        "resize": 1024,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf):
        super().__init__(**conf)  # Update with default configuration.
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.conf.descriptor_dim, kernel_size=1, stride=1, padding=0
        )

        url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"  # noqa
        self.load_state_dict(torch.hub.load_state_dict_from_url(url))

        if self.conf.max_num_keypoints is not None and self.conf.max_num_keypoints <= 0:
            raise ValueError("max_num_keypoints must be positive or None")

    def forward(self, data: dict) -> dict:
        """Compute keypoints, scores, descriptors for image"""
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        image = data["image"]

        input_image = image

        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)

        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        score_map = scores
        scores = simple_nms(scores, self.conf.nms_radius, self.conf.nms_supp)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        best_kp = torch.where(scores > self.conf.detection_threshold)
        scores = scores[best_kp]

        # print(f"best_kp size {scores.shape}")

        # Separate into batches
        keypoints = [
            torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)
        ]
        scores = [scores[best_kp[0] == i] for i in range(b)]

        # Keep the k keypoints with highest score
        if self.conf.max_num_keypoints is not None:
            keypoints, scores = list(
                zip(
                    *[
                        top_k_keypoints(k, s, self.conf.max_num_keypoints)
                        for k, s in zip(keypoints, scores)
                    ]
                )
            )

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        descriptor_map = descriptors

        # Extract descriptors
        descriptors = [
            sample_descriptors(k[None], d[None], 8)[0]
            for k, d in zip(keypoints, descriptors)
        ]

        return {
            "input_image": input_image,
            "score_map": score_map,
            "descriptor_map": torch.permute(descriptor_map, (0, 2, 3, 1)).contiguous(),
            "keypoints": torch.stack(keypoints, 0),
            "keypoint_scores": torch.stack(scores, 0),
            "descriptors": torch.stack(descriptors, 0).transpose(-1, -2).contiguous(),
        }
