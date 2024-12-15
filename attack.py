import numpy as np
import argparse

import torch

import utils
import data_loader
import models
import attack_alg

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"--> Using device: {device} <--")

parser = argparse.ArgumentParser(description="attack")

num_categories_dict = {
    2: "toy",  # test toy dataset
    50: "textshape50",
    209: "human16-209",
    16: "",
    1000: "",
}

#### data
parser.add_argument(
    "data",
    metavar="DIR",
    nargs="?",
    default="",
    help="path to dataset (default: imagenet)",
)
parser.add_argument(
    "--img-folder-txt", type=str, help="path to a textfile of image folders used"
)
parser.add_argument(
    "--save-dir", default=".", type=str, help="path to save the checkpoints"
)

### model
parser.add_argument("--model-pth", type=str, help="path to a neural predictor")
parser.add_argument("--arch", default="resnet18", type=str, help="classifier arch")

parser.add_argument(
    "--num-category",
    default=50,
    type=int,
    help=f"number of categories to use, must be one of {list(num_categories_dict.keys())}",
)
parser.add_argument(
    "--category-209",
    action="store_true",
    default=True,
    #     help="use the 209 fine-grained categories belonged the 16 basic-level categories",
)
parser.add_argument(
    "--category-16", action="store_true", help="use the 16 basic-level categories"
)
parser.add_argument(
    "--category-1k", action="store_true", help="use the original 1k categories"
)

parser.add_argument(
    "--append-layer",
    default="None",
    type=str,
    help="append which layer: [default(None), bandpass, blur] to the beginning of the model",
)
parser.add_argument(
    "--kernel-size",
    default=31,
    type=int,
    help="kernel size for the bandpass/blur layer",
)
parser.add_argument(
    "--custom-sigma",
    default=None,
    type=float,
    help="custom sigma for the bandpass layer",
)

### attack params
parser.add_argument(
    "--lp", type=str, default="inf", help="[linf], [l2]: Lp norm to use for attack"
)
parser.add_argument(
    "--attack-alg",
    type=str,
    help="which attack algorithm: pgd, fgsm, deepfool, natural",
)
parser.add_argument(
    "--perturb-type",
    type=str,
    help="which perturbation type use in natural attack: noise, blur, rotation",
)
parser.add_argument("--severity", default=3, type=int, help="natural attack severity")
parser.add_argument("--seed", default=None, type=int, help="seed for attack")

parser.add_argument("--batch-size", type=int, default=128, help="val batch size")
parser.add_argument(
    "--workers", default=4, type=int, help="number of data loading workers (default: 4)"
)
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
parser.add_argument("--cpu", action="store_true", help="use CPU")


def main():
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.lp == "linf":
        EPSILON_LS = (np.arange(5) + 1) / 255
        utils.print_safe(
            f"Using Linf norm, Epsilon to be tested: \n{EPSILON_LS}", flush=True
        )
    else:
        EPSILON_LS = []
        utils.print_safe(
            f"Using L2 norm, Epsilon to be tested: \n{EPSILON_LS}", flush=True
        )

    ## -- save dir
    utils.print_safe(f"******* Saving to: {args.save_dir}")
    utils.make_directory(args.save_dir)

    ## -- data
    _, val_loader, _, val_sampler = data_loader.build_data_loader(args)
    utils.print_safe(f"Data loaded: val: {len(val_loader)}", flush=True)

    ## -- create model
    # if args.category_16:
    #     num_classes = 16
    # elif args.category_1k:
    #     num_classes = 1000
    # else:
    #     num_classes = 209
    classifier = models.get_classifier(
        args.arch, num_classes=args.num_category, pretrained=False
    )

    if args.append_layer == "bandpass":
        model = models.BandPassNet(
            classifier, kernel_size=args.kernel_size, custom_sigma=args.custom_sigma
        )
    elif args.append_layer == "blur":
        model = models.BlurNet(classifier, sigma=args.custom_sigma)
        utils.print_safe(f"Using blur layer, sigma: {args.custom_sigma}, "
                         f"thus kernel size is: {model.kernel_size}.\n"
                         f"\t!!IGNORING the args.kernel_size argument!!")
        # raise NotImplementedError
        # model = models.BlurNet(classifier)
    else:
        model = classifier
    checkpoint = torch.load(args.model_pth, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model = models.AttackNet(model, args.append_layer)
    model = model.to(device).eval()
    utils.print_safe(model)

    if args.attack_alg == "natural":
        results = []
        perturb_types = [
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
        for perturb in perturb_types:
            results_per_type = {}
            results_per_type["type"] = perturb
            results_per_type["plot"] = []
            for severity in range(0, 6):
                print(perturb, severity)
                original_acc_sum, perturb_accs, n = attack_alg.natural_attack(
                    val_loader,
                    model,
                    device,
                    severity=severity,
                    perturbation=perturb,
                )
                utils.print_safe(
                    f"clean accuracy:  {(original_acc_sum / n * 100):.2f}, "
                    f"perturbed accuracy: {(perturb_accs / n * 100):.2f}"
                )
                if severity == 0:
                    result = {
                        "severity": severity,
                        "clean_acc": original_acc_sum / n * 100,
                        "perturb_acc": original_acc_sum / n * 100,
                    }
                else:
                    result = {
                        "severity": severity,
                        "clean_acc": original_acc_sum / n * 100,
                        "perturb_acc": perturb_accs / n * 100,
                    }
                results_per_type["plot"].append(result)

            results.append(results_per_type)
        utils.pickle_dump(
            results,
            f"{args.save_dir}/{args.arch}-layer-{args.append_layer}-attk-{args.attack_alg}.pkl",
        )
    else:
        results = {"args": args, "eps": EPSILON_LS, "clean_acc": [], "perturb_acc": []}
        for i, epsilon in enumerate(EPSILON_LS):
            print(
                f"\n-> Current Epsilon [{i}]/[{len(EPSILON_LS)}]: {epsilon:4f} ({epsilon * 255:.0f}/255)"
            )

            original_acc_sum, perturb_accs, n = attack_alg.foolbox_attack(
                val_loader, model, device, epsilon, args.lp, args.attack_alg
            )

            utils.print_safe(
                f"clean accuracy:  {(original_acc_sum / n * 100):.2f}, "
                f"perturbed accuracy: {(perturb_accs / n * 100):.2f}"
            )

            results["clean_acc"].append(original_acc_sum / n * 100)
            results["perturb_acc"].append(perturb_accs / n * 100)

        utils.pickle_dump(
            results,
            f"{args.save_dir}/{args.arch}-layer-{args.append_layer}-attk-{args.lp}-{args.attack_alg}.pkl",
        )


if __name__ == "__main__":
    args = parser.parse_args()

    args.distributed = False
    args.train_workers = args.workers
    args.test_workers = args.workers

    utils.show_input_args(args)

    main()
