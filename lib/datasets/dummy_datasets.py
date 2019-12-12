# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.collections import AttrDict


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_maze_dataset():
    ds = AttrDict()
    classes = ['__background__','signage','traffic_sign','traffic_light']
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_apollo_dataset():
    ds = AttrDict()
    classes = ['__background__','ignore','pedestrian','motorcyclist','car','bus','truck','tricyclelist','van','cyclist','trafficcone','barrowlist']
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_apollo_laneseg_dataset():
    ds = AttrDict()
    classes = ['__background__','s_w_d','s_y_d','ds_w_dn','ds_y_dn','sb_w_do','sb_y_do','b_w_g',
    'b_y_g','db_w_g','db_y_g','db_w_s','s_w_s','ds_w_s','s_w_c','s_y_c','s_w_p','s_n_p','c_wy_z',
    'a_w_u','a_w_t','a_w_tl','a_w_tr','a_w_tlr','a_w_l','a_w_r','a_w_lr','a_n_lu','a_w_tu','a_w_m',
    'a_y_t','b_n_sr','d_wy_za','r_wy_np','vom_wy_n','om_n_n']
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds