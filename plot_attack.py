import pickle
import matplotlib.pyplot as plt

with open("./resnet18-layer-None-attk-natural.pkl", "rb") as f:
    none_results = pickle.load(f)

with open("./resnet18-layer-bandpass-attk-natural.pkl", "rb") as f:
    bandpass_results = pickle.load(f)
results_list = [none_results, bandpass_results]
results_label = ["baseline", "bandpass"]
plot_types = perturb_types = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    # "defocus_blur",
    # "glass_blur",
    # "motion_blur",
    # "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]
for plot_type in plot_types:
    plt.figure(figsize=(10, 6))

    for i, results in enumerate(results_list):
        plot_info = [r["plot"] for r in results if r["type"] == plot_type][0]
        severities = [info["severity"] for info in plot_info]
        accuracies = [info["perturb_acc"] for info in plot_info]
        plt.plot(severities, accuracies, marker="o", label=results_label[i])

    plt.title(f"{plot_type}")
    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.legend(title="Type")
    # plt.grid(False)
    plt.savefig(f"{plot_type}")
