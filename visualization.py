import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchmodels
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from models import BandPassNet, get_classifier, BlurNet
import torch.nn.functional as F

def load_model(model_type="BandPassNet", weight_path=None, num_classes=50, kernel_size=31, sigma=1.5):
    classifier = get_classifier("resnet18", num_classes, pretrained=True)

    if model_type == "BandPassNet":
        model = BandPassNet(classifier, kernel_size, sigma)
    elif model_type == "BlurNet":
        model = BlurNet(classifier, sigma)
    else:
        model = classifier

    if weight_path:
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Successfully loaded weights from {weight_path}")
    return model


activation = {}

def hook_fn(name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook

def register_hooks(model, target_layers=None):
    for name, module in model.named_modules():
        if (name in target_layers):
            module.register_forward_hook(hook_fn(name))

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def plot_freq_amplitude(im):
    h, w = im.shape
    y, x = np.indices((h, w))
    center = (h//2, w//2)

    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    radial_mean = np.bincount(radius.ravel(), weights=im.ravel()) / np.bincount(radius.ravel())
    radial_mean = radial_mean[:min(im.shape) // 2]
    plt.plot(radial_mean)
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Amplitude')
    plt.title("Radial Frequency Amplitude Distribution")


def analyze_spectrum_with_radial_mean(magnitude_spectrum):
    h, w = magnitude_spectrum.shape
    y, x = np.indices((h, w))
    center = (h // 2, w // 2)

    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    radial_mean = np.bincount(radius.ravel(), weights=magnitude_spectrum.ravel()) / np.bincount(radius.ravel())

    total_energy = np.sum(radial_mean)
    high_freq_energy = np.sum(radial_mean[80:])  # High frequencies beyond radius 80
    low_freq_energy = np.sum(radial_mean[:20])  # Low frequencies within radius 20

    high_freq_ratio = high_freq_energy / total_energy
    low_freq_ratio = low_freq_energy / total_energy

    return high_freq_ratio, low_freq_ratio


def visualize_image_and_spectrum(input_tensor):
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.ravel()
    for i in range(3):
        # Input Image Channel
        axs[i].imshow(input_tensor[0, i].cpu().numpy(), cmap='gray')
        axs[i].set_title(f"Input Image Channel {i}")
        axs[i].axis('off')

        # Spectrum
        freq_map = np.fft.fftshift(np.fft.fft2(input_tensor[0, i].cpu().numpy()))
        magnitude_spectrum = np.log(np.abs(freq_map) + 1e-10)
        axs[i + 3].imshow(magnitude_spectrum, cmap="gray")
        axs[i + 3].set_title(f"Channel {i} Frequency Spectrum")
        axs[i + 3].axis('off')

        # Radial Frequency Distribution (using `plot_freq_amplitude` logic)
        h, w = magnitude_spectrum.shape
        y, x = np.indices((h, w))
        center = (h // 2, w // 2)

        radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        radial_mean = np.bincount(radius.ravel(), weights=magnitude_spectrum.ravel()) / np.bincount(radius.ravel())
        radial_mean = radial_mean[:min(magnitude_spectrum.shape) // 2]

        axs[i + 6].plot(radial_mean, label="Radial Mean")
        axs[i + 6].set_title(f"Radial Frequency Distribution: Channel {i}")
        axs[i + 6].set_xlabel('Spatial Frequency')
        axs[i + 6].set_ylabel('Amplitude')
        axs[i + 6].legend()

    plt.tight_layout()
    plt.show()




def load_model(model_type="BandPassNet", weight_path=None, num_classes=50, kernel_size=31, sigma=1.5):
    classifier = get_classifier("resnet18", num_classes, pretrained=True)

    if model_type == "BandPassNet":
        model = BandPassNet(classifier, kernel_size, sigma)
    elif model_type == "BlurNet":
        model = BlurNet(classifier, sigma)
    else:
        model = classifier

    if weight_path:
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Successfully loaded weights from {weight_path}")
    return model



activation = {}

def hook_fn(name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook

def register_hooks(model, target_layers=None):
    for name, module in model.named_modules():
        if (name in target_layers):
            module.register_forward_hook(hook_fn(name))



def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor



def plot_freq_amplitude(im):
    h, w = im.shape
    y, x = np.indices((h, w))
    center = (h//2, w//2)

    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    radial_mean = np.bincount(radius.ravel(), weights=im.ravel()) / np.bincount(radius.ravel())
    radial_mean = radial_mean[:min(im.shape) // 2]
    plt.plot(radial_mean)
    plt.xlabel('Spatial Frequency')
    plt.ylabel('Amplitude')
    plt.title("Radial Frequency Amplitude Distribution")
    return radial_mean


def analyze_spectrum_with_radial_mean(magnitude_spectrum):
    high_thres = int(80*len(magnitude_spectrum)/224)
    low_thres = int(16*len(magnitude_spectrum)/224)
    # print(len(magnitude_spectrum))
    h, w = magnitude_spectrum.shape
    y, x = np.indices((h, w))
    center = (h // 2, w // 2)
    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)

    radial_mean = np.bincount(radius.ravel(), weights=magnitude_spectrum.ravel()) / np.bincount(radius.ravel())
    # radial_mean = (radial_mean - radial_mean.min()) / (radial_mean.max() - radial_mean.min())
    radial_mean -= radial_mean.min()
    total_energy = np.sum(radial_mean)
    high_freq_energy = np.sum(radial_mean[high_thres:])  # High frequencies beyond radius 80
    low_freq_energy = np.sum(radial_mean[:low_thres])  # Low frequencies within radius 20

    high_freq_ratio = high_freq_energy / total_energy
    low_freq_ratio = low_freq_energy / total_energy

    return high_freq_ratio, low_freq_ratio

def analyze_average_response():
    for layer_name, feature_map in activation.items():
        # Upscale Feature Maps for Better Visualization
        # upscaled_feature_map = F.interpolate(feature_map, size=(224, 224), mode='bilinear', align_corners=False)
        upscaled_feature_map = feature_map
        num_channels = upscaled_feature_map.shape[1]
        high_freq_ratios = []
        low_freq_ratios = []

        for i in range(num_channels):
            spatial_response = upscaled_feature_map[0, i].cpu().numpy()
            freq_map = np.fft.fftshift(np.fft.fft2(spatial_response))
            magnitude_spectrum = np.log(np.abs(freq_map) + 1e-16)

            high_ratio, low_ratio = analyze_spectrum_with_radial_mean(magnitude_spectrum)
            high_freq_ratios.append(high_ratio)
            low_freq_ratios.append(low_ratio)

        print(f"Layer: {layer_name}")
        print(f"  Avg High-Frequency Ratio: {np.mean(high_freq_ratios):.2f}")
        print(f"  Avg Low-Frequency Ratio: {np.mean(low_freq_ratios):.2f}")

        average_spatial_response = upscaled_feature_map.mean(dim=1, keepdim=True).squeeze(0).squeeze(0).cpu().numpy()
        plt.figure(figsize=(10, 8))
        plt.imshow(average_spatial_response, cmap="rainbow")
        plt.title(f"Average Spatial Response: {layer_name}")
        plt.colorbar()
        plt.axis("off")
        plt.show()

        freq_map = np.fft.fftshift(np.fft.fft2(average_spatial_response))
        magnitude_spectrum = np.log(np.abs(freq_map) + 1e-10)
        plt.figure(figsize=(10, 8))
        plt.imshow(magnitude_spectrum, cmap="gray")
        plt.title(f"Average Frequency Response: {layer_name}")
        plt.colorbar()
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 8))
        radial_mean = plot_freq_amplitude(magnitude_spectrum)
        plt.title(f"Radial Frequency Amplitude: {layer_name}")
        plt.show()

def visualize_activations_and_frequency(channels_per_row=6):
    for layer_name, feature_map in activation.items():
        # num_channels = feature_map.shape[1]
        num_channels = min(12, feature_map.shape[1])
        num_rows = -(-num_channels // channels_per_row)  # Calculate rows needed

        fig, axs = plt.subplots(num_rows, channels_per_row, figsize=(20, num_rows * 5))
        fig.suptitle(f"Layer: {layer_name}")

        for i in range(num_channels):
            row, col = divmod(i, channels_per_row)
            ax = axs[row, col] if num_rows > 1 else axs[col]

            ax.imshow(feature_map[0, i].cpu().numpy(), cmap="rainbow")
            ax.axis("off")
            ax.set_title(f"Channel {i}")

        plt.tight_layout()
        plt.show()

def analyze_input_mean_channel(input_tensor):
    mean_channel = input_tensor[0].mean(dim=0).cpu().numpy()  # 平均三个通道

    freq_map = np.fft.fftshift(np.fft.fft2(mean_channel))  # 计算FFT并中心化
    magnitude_spectrum = np.log(np.abs(freq_map) + 1e-10)  # 取对数幅度谱

    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title(f"Average Frequency Response: Input Image")
    plt.colorbar()
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(10, 8))
    radial_mean = plot_freq_amplitude(magnitude_spectrum)
    plt.title(f"Radial Frequency Amplitude: Input Image")
    plt.show()

    plt.tight_layout()
    plt.show()

    high_ratio, low_ratio = analyze_spectrum_with_radial_mean(magnitude_spectrum)
    print(f"Mean Channel: High-Frequency Ratio: {high_ratio:.2f}, Low-Frequency Ratio: {low_ratio:.2f}")

if __name__ == "__main__":
    MODEL_TYPE = "BlurNet"  # Options: "BandPassNet", "BlurNet", "ResNet"
    WEIGHT_PATH = "ckpt_epk40_blur.pth"
    IMAGE_PATH = "n02123159_667.JPEG"
    # TARGET_LAYERS = ["conv_real", "blur_layer","classifier.conv1", "classifier.layer1.0.conv1", "classifier.layer2.0.conv1", "classifier.layer3.0.conv1"]
    TARGET_LAYERS = ["classifier.layer2.0.conv1", "classifier.layer3.0.conv1"]
    model = load_model(model_type=MODEL_TYPE, weight_path=WEIGHT_PATH, sigma=1.5)
    model.eval()
    print("\n".join([name for name, _ in model.named_modules()]))

    register_hooks(model, TARGET_LAYERS)

    transform = get_transform()
    input_tensor = load_image(IMAGE_PATH, transform)
    with torch.no_grad():
        output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1)
    print(f"Predicted class index: {predicted_class.item()}")
    visualize_image_and_spectrum(input_tensor)

    visualize_activations_and_frequency()

    # # Analyze Spectrum Ratios
    # for i in range(3):
    #     freq_map = np.fft.fftshift(np.fft.fft2(input_tensor[0, i].cpu().numpy()))
    #     magnitude_spectrum = np.log(np.abs(freq_map) + 1e-10)
    #     high_ratio, low_ratio = analyze_spectrum_with_radial_mean(magnitude_spectrum)
    #     print(f"Channel {i}: High-Frequency Ratio: {high_ratio:.2f}, Low-Frequency Ratio: {low_ratio:.2f}")
    analyze_input_mean_channel(input_tensor)

    analyze_average_response()