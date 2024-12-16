import pickle
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

with open("./new_resnet18-layer-None-attk-natural.pkl", "rb") as f:
    none_results = pickle.load(f)

with open("./new_resnet18-layer-bandpass-attk-natural.pkl", "rb") as f:
    bandpass_results = pickle.load(f)

with open("./resnet18-layer-bandpass-2.0-attk-natural.pkl", "rb") as f:
    bandpass_human_results = pickle.load(f)

with open("./resnet18-layer-blur-1.5-attk-natural.pkl", "rb") as f:
    blur_15_results = pickle.load(f)
with open("./resnet18-layer-blur-4.0-attk-natural.pkl", "rb") as f:
    blur_40_results = pickle.load(f)
results_list = [
    none_results,
    bandpass_results,
    bandpass_human_results,
    blur_15_results,
    blur_40_results,
]
results_label = ["baseline", "bandpass", "human", "blur15", "blur40"]
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
    # bax = brokenaxes(ylims=((0, 15), (60, 100)), hspace=0.1)
    for i, results in enumerate(results_list):
        plot_info = [r["plot"] for r in results if r["type"] == plot_type][0]
        severities = [info["severity"] for info in plot_info]
        accuracies = [info["perturb_acc"] for info in plot_info]
        # bax.plot(severities, accuracies, marker="o", label=results_label[i])
        plt.plot(severities, accuracies, marker="o", label=results_label[i])

    # bax.set_title(f"{plot_type}")
    # bax.set_xlabel("Severity")
    # bax.set_ylabel("Accuracy")
    # bax.legend(title="Type")
    plt.title("Accuracy vs Severity")
    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.legend(title="Type")
    # plt.grid(False)
    plt.savefig(f"plot/{plot_type}")
