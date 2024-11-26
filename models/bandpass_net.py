import torch
import torch.nn as nn
# import torchvision
# from collections import OrderedDict
import numpy as np
from . import filter


def construct_bandpass_filter(kernel_size):
    x_freqs = np.linspace(0, kernel_size, kernel_size)
    x_freqs = (x_freqs / kernel_size) * 224.  # convert to cycles per image, 224 is known for imagenet, need to change for other imagesets
    gauss_fit, fil_recon = filter.human_filter((kernel_size, kernel_size), x_freqs)

    fil_recon = np.fft.ifftshift(fil_recon)
    spatial_filter = np.fft.ifft2(fil_recon)
    spatial_filter = np.fft.ifftshift(spatial_filter)
    spatial_filter_real = np.real(spatial_filter)
    spatial_filter_imag = np.imag(spatial_filter)

    spatial_filter_real = torch.tensor(np.flip(spatial_filter_real).copy(), dtype=torch.float32)
    spatial_filter_imag = torch.tensor(np.flip(spatial_filter_imag).copy(), dtype=torch.float32)

    return gauss_fit, fil_recon, spatial_filter_real, spatial_filter_imag


class BandPassNet(nn.Module):
    def __init__(self, classifier, kernel_size):
        """
        :param classifier:
        """
        super(BandPassNet, self).__init__()
        self.classifier = classifier
        in_channels = self.classifier.conv1.in_channels

        self.conv_real = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding='same', padding_mode='reflect',
                                   bias=False)
        self.conv_imag = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding='same', padding_mode='reflect',
                                   bias=False)
        self.gauss_fit, self.fil_recon, self.spatial_filter_real = self._init_filter(kernel_size, in_channels)

    def _init_filter(self, kernel_size, num_channel):
        gauss_fit, fil_recon, \
            spatial_filter_real, spatial_filter_imag = construct_bandpass_filter(kernel_size)

        with torch.no_grad():
            self.conv_real.weight.zero_()
            self.conv_imag.weight.zero_()
            for i in range(num_channel):
                self.conv_real.weight[i, i] = spatial_filter_real
                self.conv_imag.weight[i, i] = spatial_filter_imag
            
            self.conv_real.weight.requires_grad = False
            self.conv_imag.weight.requires_grad = False

        return gauss_fit, fil_recon, spatial_filter_real
    
    def forward(self, x, filter_only=False):
       
        x = self.conv_real(x)
        # imag_part = self.conv_imag(x)
        # phase = torch.atan2(imag_part, real_part)
        # x = torch.sqrt(real_part ** 2 + imag_part ** 2)

        if filter_only:
            return x
        x = self.classifier(x)
        return x


def calc_contrast(im):
    return np.std(np.asarray(im))


def normalize_to_original(original_img, filtered_img):
    # Compute the mean and standard deviation of the filtered image
    axis = (0, 1)
    original_contrast = np.std(original_img,axis=axis)

    # Compute the filtered image's RMS contrast (std deviation)
    filtered_contrast = np.std(filtered_img, axis=axis)

    # Scale filtered image to match original contrast
    contrast_scaled_img = (filtered_img - np.mean(filtered_img, axis=axis)) * (original_contrast / filtered_contrast) + np.mean(filtered_img, axis=axis)
    return contrast_scaled_img