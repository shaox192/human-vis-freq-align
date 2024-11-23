
import torch.distributed as dist
import os
import pickle as pkl
import cv2


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
