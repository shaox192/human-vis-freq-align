
import torch.nn as nn


class AttackNet(nn.Module):
    def __init__(self, model, append_layer):
        super(AttackNet, self).__init__()
        self.module = model
        self.append_layer = append_layer

    def forward(self, x):
        if self.append_layer == "bandpass":
            return self.module(x, filter_only=False)
        else:
            return self.module(x)
    