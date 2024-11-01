import matplotlib.pyplot as plt

# from skimage import data, filters

# image = data.camera()

# cutoff frequencies as a fraction of the maximum frequency
cutoffs = [0.02, 0.08, 0.16]


# def DOB_filter():
#     """
#     Difference of Butterworth
#     """

#     lower_pass = filters.butterworth(image, cutoff_frequency_ratio=0.02, order=3.0, high_pass=False)
#     upper_pass = filters.butterworth(image, cutoff_frequency_ratio=0.1, order=3.0, high_pass=False)

#     dob = upper_pass - lower_pass

#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))

#     axes[0].imshow(lower_pass, cmap='gray')
#     axes[0].set_title('lowpass')
#     axes[1].imshow(upper_pass, cmap='gray')
#     axes[1].set_title('highpass')
#     axes[2].imshow(dob, cmap='gray')
#     axes[2].set_title('difference of Butterworth')

#     plt.show()


import numpy as np
import matplotlib.pyplot as plt

def butterworth_filter(shape, cutoff, order, high_pass=False):
    # https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/12cab215afbbe6694402d8d6458ce76f_MITRES_6_007S11_lec24.pdf
    h, w = shape
    # grid of (u, v) coordinates

    u = np.arange(-w//2, w//2)
    v = np.arange(-h//2, h//2)
    U, V = np.meshgrid(u, v)

    D = np.sqrt(U**2 + V**2)  # Distance from the center
    
    # Calculate Butterworth filter
    if high_pass:
        H = 1 - 1 / (1 + (D / cutoff)**(2 * order))
    else:
        H = 1 / (1 + (D / cutoff)**(2 * order))
    return H


def gaussian_filter(shape, cutoff, high_pass=False):
    h, w = shape
    u = np.arange(-w//2, w//2)
    v = np.arange(-h//2, h//2)
    U, V = np.meshgrid(u, v)

    D = np.sqrt(U**2 + V**2)
    if high_pass:
        H = 1 - np.exp(-D**2 / (2 * cutoff**2))
    else:
        H = np.exp(-D**2 / (2 * cutoff**2))
    return H


def DoB_filter(shape, cutoff_low, cutoff_high, order_low, order_high):
    low = butterworth_filter(shape, cutoff_low, order_low)
    high = butterworth_filter(shape, cutoff_high, order_high)
    return high - low

def DoG_filter(shape, sigma_low, sigma_high):
    low = gaussian_filter(shape, sigma_low)
    high = gaussian_filter(shape, sigma_high)
    return high - low


def get_image_freq(image):
    # Compute the 2D Fourier Transform of the input image
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)
    mag_spectrum = np.log(np.abs(Fshift))
    return mag_spectrum


def filter_image_infreq(image, filter):
    # Compute the 2D Fourier Transform of the input image
    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)

    # Apply the filter
    Fshift = Fshift * filter
    filtered_spectrum = np.log(np.abs(Fshift))

    # Compute the inverse 2D Fourier Transform
    F = np.fft.ifftshift(Fshift)
    img_back = np.fft.ifft2(F)
    img_back = np.abs(img_back)
    return img_back, filtered_spectrum

