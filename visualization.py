import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchmodels
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models import BandPassNet, get_classifier, BlurNet

# ================================
# 1. 加载模型
# ================================
def load_model():
    # 配置参数
    num_classes = 50  # 确保与训练时一致
    kernel_size = 31
    custom_sigma = 1.5
    custom_sigma = 0.424619

    # 加载基础 ResNet 模型
    classifier = get_classifier("resnet18", num_classes, pretrained=True)

    # 包装成 BandPassNet
    model = BandPassNet(classifier, kernel_size, custom_sigma)
    # model = BlurNet(classifier, custom_sigma)
    model = classifier
    weight_path = "ckpt_epk40_none.pth"
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))  # 根据需求调整设备
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"Successfully loaded weights from {weight_path}")

    return model


activation = {}

def hook_fn(name):
    def hook(module, input, output):
        activation[name] = output.detach()
    return hook

def register_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):  # 只注册卷积层
            if name == "conv_real":
                module.register_forward_hook(hook_fn(name))
        # if name == "classifier.layer4":  # ResNet 的最后一个卷积模块
        #     module.register_forward_hook(hook_fn("layer4"))
        # elif name == "classifier.avgpool":  # 全局平均池化之前
        #     module.register_forward_hook(hook_fn("avgpool"))

def get_transform():
    return transforms.Compose([  # 调整为与训练一致的尺寸
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # 增加 batch 维度
    return input_tensor

def visualize_activations():
    channels = 8
    for layer_name, feature_map in activation.items():
        num_channels = feature_map.shape[1]  # 通道数
        fig, axs = plt.subplots(1, min(num_channels, channels), figsize=(20, 5))  # 每层显示前 8 个通道
        fig.suptitle(f"Layer: {layer_name}")

        for i in range(min(num_channels, channels)):
            axs[i].imshow(feature_map[0, i].cpu().numpy(), cmap="rainbow")
            axs[i].axis("off")
        plt.show()

if __name__ == "__main__":
    model = load_model()
    model.eval()

    register_hooks(model)

    image_path = "n02123159_667.JPEG"  # 替换为实际图片路径
    transform = get_transform()
    input_tensor = load_image(image_path, transform)

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)
    print(torch.argmax(output, dim=1))

    # 可视化特征图
    visualize_activations()