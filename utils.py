
import torch.distributed as dist
import os
import pickle as pkl
import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import entropy


def read_image(image_path, gray=False, to_float=True, imagenet_transform=True):
    """
    Read an image from a file path using OpenCV
    :param image_path: path to the image file
    :return: the image as a numpy array
    """
    # read the image
    if gray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if imagenet_transform:
        img = resize_cv2(img)
        img = crop_cv2(img)
        
    if to_float:
        img = img.astype(float)
        img /= 255.0

    return img


def resize_cv2(image, target_size=256):
    h, w = image.shape[:2]
    if h < w:
        new_h, new_w = target_size, int(target_size * w / h)
    else:
        new_h, new_w = int(target_size * h / w), target_size
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image

def crop_cv2(image, crop_size=224):
    h, w = image.shape[:2]
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    cropped_image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    return cropped_image


def pickle_dump(data, fpth):
    print(f"writing to: {fpth}")
    with open(fpth, 'wb') as f:
        pkl.dump(data, f)


def pickle_load(fpth):
    print(f"loading from: {fpth}")
    with open(fpth, 'rb') as f:
        return pkl.load(f)


def show_input_args(args):
    print("\n***check params ---------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("--------------------------\n")



################## Training Utils ##################

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def print_safe(print_str, **kwargs):
    if "flush" not in kwargs:
        kwargs["flush"] = True
    if is_main_process():
        print(print_str, **kwargs)

def make_directory(pth):
    if is_main_process():
        if not os.path.exists(pth):
            print(f"Making output dir at {pth}")
            os.makedirs(pth, exist_ok=True)
        else:
            print(f"Path {pth} exists.")



################## FREQUENCY UTILS ##################
HUMAN_AVG_GAUSS = [4.0, 4.5, 0.424619]

def fit_gaussian(data, A, mu, sigma, convert_data=False):
    """
    Fits a gaussian to the data
    """
    def gauss(x):
        return A * np.exp(-(x-mu)**2/(2.*sigma**2))
    
    data2fit = np.array(data)
    if convert_data:
        data2fit = np.log2(data2fit / 1.75)

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