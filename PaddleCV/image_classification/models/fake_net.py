from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}

class FakeNet():
    """
        \brief      Fake network returning tensor of input shape filled with zeros.
    """

    def __init__(self):
        self.params = train_parameters

    def net(self, input, class_dim=1000):
        return fluid.layers.fill_constant_batch_size_like(input, [1, class_dim], "float32", 0.0)
