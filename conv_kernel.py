
import torch
import torch.nn as nn

import numpy as np


class Kernel(nn.Module):
    def __init__(self, freq_filter, size, num_channel):
        """
        freq_filter: filter in the frequency domain
        size: size of conv kernel
        """
        super(Kernel, self).__init__()

        self.spatial_filter = self._init_filter(freq_filter, size, num_channel)
    
    def _init_filter(self, freq_filter, size, num_channel):
        # half_size = size // 2
        # spatial_filter = np.abs(np.fft.ifftshift(np.fft.ifft2(freq_filter)))
        # #spatial_filter /= spatial_filter.sum()
        # spatial_filter = torch.tensor(spatial_filter)
        # spatial_filter = torch.flip(spatial_filter, dims=[0, 1])
        # print(spatial_filter)
        # conv = nn.Conv2d(num_channel, num_channel, size, padding='same', padding_mode='circular', bias=False)
        # conv.weight.data *= 0
        # for i in range(num_channel):
        #     conv.weight.data[i, i] = spatial_filter
        # conv.weight.requires_grad = False
        spatial_filter = np.fft.ifft2(freq_filter)
        spatial_filter = np.fft.ifftshift(spatial_filter)
        spatial_filter_real = np.real(spatial_filter)
        spatial_filter_imag = np.imag(spatial_filter)

        spatial_filter_real = torch.tensor(np.flip(spatial_filter_real).copy(), dtype=torch.float32)
        spatial_filter_imag = torch.tensor(np.flip(spatial_filter_imag).copy(), dtype=torch.float32)

        self.conv_real = nn.Conv2d(num_channel, num_channel, size, padding='same', padding_mode='reflect',
                                   bias=False)
        self.conv_imag = nn.Conv2d(num_channel, num_channel, size, padding='same', padding_mode='reflect',
                                   bias=False)
        with torch.no_grad():
            self.conv_real.weight.zero_()
            self.conv_imag.weight.zero_()
            for i in range(num_channel):
                self.conv_real.weight[i, i] = spatial_filter_real
                self.conv_imag.weight[i, i] = spatial_filter_imag
            self.conv_real.weight.requires_grad = False
            self.conv_imag.weight.requires_grad = False

        return spatial_filter_real


    def forward(self, x):
        real_part = self.conv_real(x)
        imag_part = self.conv_imag(x)

        output = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.atan2(imag_part, real_part)
        return output


def conv_kernel(freq_filter, size, num_channel):
    model = Kernel(freq_filter, size, num_channel)
    print(model)

    model = model.double()
    return model, model.spatial_filter


