import numpy as np
import utils


def butterworth_filter(shape, cutoff, order, high_pass=False):
    # https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/12cab215afbbe6694402d8d6458ce76f_MITRES_6_007S11_lec24.pdf
    h, w = shape
    # grid of (u, v) coordinates

    u = np.arange(-(w//2), -(w//2) + shape[1])    
    v = np.arange(-(h//2), -(h//2) + shape[1])    
    U, V = np.meshgrid(u, v)

    D = np.sqrt(U**2 + V**2)  # Distance from the center
    
    # Calculate Butterworth filter
    if high_pass:
        H = 1 - 1 / (1 + (D / cutoff)**(2 * order))
    else:
        H = 1 / (1 + (D / cutoff)**(2 * order))
    return H


def gaussian_filter(shape, sigma, high_pass=False):
    h, w = shape
    u = np.arange(-(w//2), -(w//2) + shape[1])    
    v = np.arange(-(h//2), -(h//2) + shape[1])   
    U, V = np.meshgrid(u, v)

    D = np.sqrt(U**2 + V**2)
    if high_pass:
        H = 1 - np.exp(-D**2 / (2 * sigma**2))
    else:
        H = np.exp(-D**2 / (2 * sigma**2))
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


def human_filter(im_shape, x_freqs):
    A, mu, sigma = utils.HUMAN_AVG_GAUSS
    gauss_fit = utils.fit_gaussian(x_freqs, A, mu, sigma, convert_data=True)
    gauss_fit = gauss_fit / gauss_fit.max()  # normalize

    fil_recon = utils.radialmean2filter(gauss_fit, im_shape)
    return gauss_fit, fil_recon

