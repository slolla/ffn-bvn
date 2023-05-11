# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils 

import os
import numpy as np

from drqv2_crff_dir import utils, clff_modules


from .env_helpers import get_env

class Encoder(nn.Module):
    def __init__(self, obs_shape, fourier_features=None, scale=None, rff=False):
        super().__init__()

        # assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.rff = rff

        if self.rff:
             self.convnet = nn.Sequential(clff_modules.CLFF(obs_shape, fourier_features, scale=scale),
                                          nn.Conv2d(fourier_features, 32, 2, stride=2),
                                          nn.ReLU(), nn.Conv2d(32, 32, 1, stride=1),
                                          nn.ReLU(), nn.Conv2d(32, 32, 1, stride=1),
                                          nn.ReLU())
            #self.convnet = nn.Sequential(clff_modules.CLFF(obs_shape, fourier_features, scale=scale),
            #                             nn.ReLU())
        else:
            self.convnet = nn.Sequential(nn.Conv2d(obs_shape, 32, 3, stride=2),
                                        nn.ReLU(), nn.Conv1d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv1d(32, 32, 3, stride=1),
                                        nn.ReLU(), nn.Conv1d(32, 32, 3, stride=1),
                                        nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        print("fourier features generated")
        h = h.view(h.shape[0], -1)
        print(h.shape)
        return h