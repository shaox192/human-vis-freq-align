import pickle
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes

with open("./new_resnet18-layer-None-attk-natural.pkl", "rb") as f:
    none_results = pickle.load(f)

with open("./new_resnet18-layer-bandpass-attk-natural.pkl", "rb") as f:
    bandpass_results = pickle.load(f)

with open("./resnet18-layer-bandpass-humanfil2-attk-natural.pkl", "rb") as f:
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
    # blur_40_results,
]
results_label = ["bl", "hcbp", "hcbe", "blur"]
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
clr_palette = {"bl": "black", "blur": "blue", "hcbp": "red", "hcbe": "purple"}
plot_lb = {"bl": "Baseline", "blur": "Gaussian-blur", "hcbp": "HC-BP", "hcbe": "HC-BE"}

fig, axes = plt.subplots(3, 5, figsize=(25, 15))
axes = axes.flatten()
for idx, plot_type in enumerate(plot_types):
    ax = axes[idx]
    # bax = brokenaxes(ylims=((0, 15), (60, 100)), hspace=0.1)
    for i, results in enumerate(results_list):
        plot_info = [r["plot"] for r in results if r["type"] == plot_type][0]
        severities = [info["severity"] for info in plot_info][1:]
        accuracies = [info["perturb_acc"] for info in plot_info][1:]
        # bax.plot(severities, accuracies, marker="o", label=results_label[i])
        color = clr_palette.get(
            results_label[i], "gray"
        )  # Default to gray if not in palette
        label = plot_lb.get(results_label[i], results_label[i])
        ax.set_xticks(range(1, 6))
        ax.plot(severities, accuracies, marker="o", color=color, label=label)

    # bax.set_title(f"{plot_type}")
    # bax.set_xlabel("Severity")
    # bax.set_ylabel("Accuracy")
    # bax.legend(title="Type")
    ax.set_title(f"{plot_type}")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Accuracy(%)")
    ax.legend(title="Type")

for idx in range(len(plot_types), len(axes)):
    fig.delaxes(axes[idx])

# Adjust spacing between subplots
plt.tight_layout()

# Save the big image
plt.savefig("plot/whole_image.png", dpi=300)
