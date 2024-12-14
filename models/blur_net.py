import torch
import torch.nn as nn
# from jarvis.models import ImageClassifier, ResNet

# Tensor = torch.Tensor

class BlurNet(nn.Module):
    def __init__(self, classifier, sigma):
        """
        :param classifier:
        sigma: 1.5 default to Li et al., 2023
        """
        super(BlurNet, self).__init__()
        self.classifier = classifier
        in_channels = self.classifier.conv1.in_channels

        #TODO: Implement blur layer
        self.sigma = sigma
        if self.sigma == 0:
            self.kernel_size = 0
            self.blur_layer = nn.Sequential() # nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', padding_mode='reflect', bias=False)
            self.blur_kernel_w = None
        else:
            self.kernel_size, self.blur_layer, self.blur_kernel_w = None, None, None
            self._init_filter(in_channels, sigma)

    def _init_filter(self, num_channel, sigma):
        
        #TODO: Implement blur layer
        ####: code from https://github.com/lizhe07/blur-net/blob/master/blurnet/models.py
        half_size = int(-(-2.5*sigma//1))
        self.kernel_size = 2*half_size+1

        x, y = torch.meshgrid(
            torch.arange(-half_size, half_size+1).to(torch.float),
            torch.arange(-half_size, half_size+1).to(torch.float),
            )
        w = torch.exp(-(x**2+y**2)/(2*sigma**2))
        w /= w.sum()

        self.blur_kernel_w = w

        self.blur_layer = nn.Conv2d(
            num_channel, num_channel, self.kernel_size,
            padding=half_size, padding_mode='circular', bias=False,
        )

        self.blur_layer.weight.data *= 0
        for i in range(num_channel):
            self.blur_layer.weight.data[i, i] = w
        self.blur_layer.weight.requires_grad = False

        return
    
    def forward(self, x, filter_only=False):
       
        x = self.blur_layer(x)

        if filter_only:
            return x
        x = self.classifier(x)
        return x
