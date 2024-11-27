import torch
import torch.nn as nn
# from jarvis.models import ImageClassifier, ResNet

# Tensor = torch.Tensor

class BlurNet(nn.Module):
    def __init__(self, classifier, kernel_size, sigma):
        """
        :param classifier:
        """
        super(BlurNet, self).__init__()
        self.classifier = classifier
        in_channels = self.classifier.conv1.in_channels

        self.sigma = sigma
        self.blur = None # nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', padding_mode='reflect', bias=False)
        self._init_filter(kernel_size, in_channels, sigma)

    def _init_filter(self, kernel_size, num_channel, sigma):
        
        #TODO: Implement blur layer
        ####: code from https://github.com/lizhe07/blur-net/blob/master/blurnet/models.py
        # half_size = int(-(-2.5*sigma//1))
        # x, y = torch.meshgrid(
        #     torch.arange(-half_size, half_size+1).to(torch.float),
        #     torch.arange(-half_size, half_size+1).to(torch.float),
        #     )
        # w = torch.exp(-(x**2+y**2)/(2*sigma**2))
        # w /= w.sum()
        # self.blur = nn.Conv2d(
        #     self.in_channels, self.in_channels, 2*half_size+1,
        #     padding=half_size, padding_mode='circular', bias=False,
        # )
        # self.blur.weight.data *= 0
        # for i in range(self.in_channels):
        #     self.blur.weight.data[i, i] = w
        # self.blur.weight.requires_grad = False
        return
    
    def forward(self, x, filter_only=False):
       
        x = self.blur(x)

        if filter_only:
            return x
        x = self.classifier(x)
        return x
