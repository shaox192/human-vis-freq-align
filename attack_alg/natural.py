import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import gaussian_blur
from foolbox import PyTorchModel, accuracy
from skimage.filters import gaussian
import os
from PIL import Image

os.environ["MAGICK_HOME"] = "/home/chengxiao/anaconda3/envs/cs543_conda"
os.environ["PATH"] += os.pathsep + os.path.join(os.environ["MAGICK_HOME"], "bin")
from imagenet_c import corrupt


def save_image(tensor_image, file_name):
    """Save a torch tensor as an image."""
    image_np = tensor_image.permute(1, 2, 0).cpu().numpy() * 255  # Convert to [0, 255]
    image_np = image_np.astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    pil_image.save(file_name)


def perturbate(images, device, severity=3, corruption_name="gaussian_noise"):
    perturbed_images = []
    for i, image in enumerate(images):
        # Convert the PyTorch image tensor to a NumPy array (H, W, C)
        image_np = image.permute(1, 2, 0).cpu().numpy() * 255  # Convert to [0, 255]
        image_np = image_np.astype(np.uint8)

        # Apply the specified corruption
        corrupted_image_np = corrupt(
            image_np, severity=severity, corruption_name=corruption_name
        )

        # Convert the corrupted NumPy array back to a PyTorch tensor
        perturbed_image = (
            torch.from_numpy(corrupted_image_np).float().div(255.0).to(device)
        )  # Normalize to [0, 1]
        perturbed_image = perturbed_image.permute(2, 0, 1)
        save_image(perturbed_image, f"perturbed_image_{i}.png")
        perturbed_images.append(perturbed_image)
    return perturbed_images


def natural_attack(val_loader, model, device, severity, perturbation, **kwargs):

    original_acc_sum = 0
    perturb_acc_sum = 0.0
    n = 0
    fmodel = PyTorchModel(model, bounds=(-3.0, 3.0))
    for images, target in val_loader:
        images, target = images.to(device), target.to(device)

        clean_acc = accuracy(fmodel, images, target)
        original_acc_sum += clean_acc * len(images)
        # Apply perturbations
        perturbed_images = perturbate(
            images, device, severity=severity, corruption_name=perturbation
        )
        perturbed_images = torch.stack(perturbed_images)
        perturb_acc = accuracy(fmodel, perturbed_images, target)
        perturb_acc_sum += perturb_acc * len(perturbed_images)
        n += len(images)

    return original_acc_sum, perturb_acc_sum, n
