
import numpy as np
import matplotlib.pyplot as plt


def plot_freq_amplitude(im):
    h, w = im.shape
    y, x = np.indices((h, w))
    center = (h//2, w//2)

    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    radial_mean = np.bincount(radius.ravel(), weights=im.ravel()) / np.bincount(radius.ravel())
    print(radial_mean.shape)
    radial_mean = radial_mean[:min(im.shape) // 2]
    plt.plot(radial_mean)
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Amplitude')
    plt.show()


def plot_spectrum(filter):
    plt.imshow(filter, cmap='gray')
    plt.colorbar()
    # plt.title(title if title else 'Filter')
    plt.show()


def im_show(image):
    fig, ax = plt.subplots()

    ax.imshow(image, cmap='gray')
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()