
import torch
import torch.nn as nn

import numpy as np

import filter
import plot_utils



class Kernel(nn.Module):
    def __init__(self, freq_filter, size, num_channel):
        """
        freq_filter: filter in the frequency domain
        size: size of conv kernel
        """
        super(Kernel, self).__init__()

        self.spatial_filter, self.conv = self._init_filter(freq_filter, size, num_channel)
    
    def _init_filter(self, freq_filter, size, num_channel):
        half_size = size // 2
        spatial_filter = np.abs(np.fft.ifftshift(np.fft.ifft2(freq_filter)))
        spatial_filter /= spatial_filter.sum()
        spatial_filter = torch.tensor(spatial_filter)

        conv = nn.Conv2d(num_channel, num_channel, size, padding=half_size, padding_mode='circular', bias=False)

        conv.weight.data *= 0
        for i in range(num_channel):
            conv.weight.data[i, i] = spatial_filter
        conv.weight.requires_grad = False

        return spatial_filter, conv


    def forward(self, x):
        return self.conv(x)


def conv_kernel(freq_filter, size, num_channel):
    model = Kernel(freq_filter, size, num_channel)
    print(model)

    model = model.double()
    return model, model.spatial_filter


