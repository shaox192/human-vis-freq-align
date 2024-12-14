import numpy as np
# import utils
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import entropy


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


def gaussian_filter(shape, sigma, equate_freq_range:tuple=None, high_pass=False):
    h, w = shape
    u = np.arange(-(w//2), -(w//2) + shape[1])
    v = np.arange(-(h//2), -(h//2) + shape[1])
    if equate_freq_range is not None:
        u = u / w * equate_freq_range[0]
        v = v / h * equate_freq_range[1]

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
    Fshift_fil = Fshift * filter

    # Compute the inverse 2D Fourier Transform
    F_fil = np.fft.ifftshift(Fshift_fil)
    img_back = np.fft.ifft2(F_fil)
    # img_back = np.abs(img_back)
    return img_back, Fshift, Fshift_fil # , filtered_spectrum

################## HUMAN FREQUENCY CHANNEL BP FILTER ##################

HUMAN_AVG_GAUSS = [4.0, 4.5, 0.424619]

def fit_gaussian(data, A, mu, sigma, convert_data=False):
    """
    Fits a gaussian to the data
    """
    def gauss(x):
        return A * np.exp(-(x-mu)**2/(2.*sigma**2))
    
    data2fit = np.array(data)
    if convert_data:
        data2fit = np.log2((data2fit / 1.75) + 1e-6)

    gauss_fit = gauss(data2fit)
    return gauss_fit

def channel_props(A, mu, sigma):
    """
    Calculates channel properties, given fit gaussian parameters
	"""
    bw = 2 * np.sqrt(np.log(4)) * sigma
    cf = 1.75 * 2 ** mu
    pns = 2**(A-4)
    
    return bw, cf, pns

def filter2radialmean(filter):
    h, w = filter.shape
    y, x = np.indices((h, w))
    center = (h//2, w//2)

    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    radial_mean = np.bincount(radius.ravel(), weights=filter.ravel()) / np.bincount(radius.ravel())

    radial_mean = radial_mean[:min(filter.shape) // 2]
    return radial_mean


def radialmean2filter(radial_mean, im_shape):
    """
    Converts radial mean to an filter
    """
    h, w = im_shape
    y, x = np.indices(im_shape)
    center = (h//2, w//2)
    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    # interpolate for smooth radial mean
    r = np.arange(len(radial_mean))
    interp_func = interp1d(r, radial_mean, kind='quadratic', 
                           bounds_error=False, fill_value="extrapolate")
    smooth_radial_mean = interp_func(radius)
    return smooth_radial_mean

def kl_divergence(p, q):
    """
    Calculates KL divergence between two distributions
    """
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()
    
    return entropy(p, q)

def mse(p, q):
    """
    Calculates mean squared error between two distributions
    """
    p = np.array(p)
    q = np.array(q)
    return np.mean((p - q)**2)


def human_filter(im_shape, x_freqs, custom_sigma=None):
    A, mu, sigma = HUMAN_AVG_GAUSS
    if custom_sigma is not None:
        sigma = custom_sigma
    print(f'\t**** Using human filter with Gaussian: A={A}, mu={mu}, sigma={sigma}')

    gauss_fit = fit_gaussian(x_freqs, A, mu, sigma, convert_data=True)
    gauss_fit = gauss_fit / gauss_fit.max()  # normalize

    fil_recon = radialmean2filter(gauss_fit, im_shape)
    return gauss_fit, fil_recon