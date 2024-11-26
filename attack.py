
import numpy as np
import argparse

import torch

import utils
import data_loader
import models
import attack_alg

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Using device: {device} <--")

parser = argparse.ArgumentParser(description='attack')

#### data
parser.add_argument('data', metavar='DIR', nargs='?', default='',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--img-folder-txt', type=str, help='path to a textfile of image folders used')

### model
parser.add_argument('--model-pth', type=str, help='path to a neural predictor')
parser.add_argument('--arch', default='resnet18', type=str, help="classifier arch")

parser.add_argument('--category-209', action='store_true', default=True,
                    help='use the 209 fine-grained categories belonged the 16 basic-level categories')
parser.add_argument('--category-16', action='store_true', help='use the 16 basic-level categories')
parser.add_argument('--category-1k', action='store_true', help='use the original 1k categories')

parser.add_argument("--append-layer", default="None", type=str, 
                    help="append which layer: [default(None), bandpass, blur] to the beginning of the model")
parser.add_argument("--kernel-size", default=31, type=int, 
                    help="kernel size for the bandpass/blur layer")

### attack params
parser.add_argument('--lp', type=str, default='inf', 
                    help='[linf], [l2]: Lp norm to use for attack')
parser.add_argument('--attack-alg', type=str,
                    help='which attack algorithm: pgd, fgsm, deepfool')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for attack')

parser.add_argument('--batch-size', type=int, default=128, help='val batch size')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')


def main():
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.lp == "linf":
        EPSILON_LS = (np.arange(5) + 1) / 255
        utils.print_safe(f"Using Linf norm, Epsilon to be tested: \n{EPSILON_LS}", flush=True)
    else:
        EPSILON_LS = []
        utils.print_safe(f"Using L2 norm, Epsilon to be tested: \n{EPSILON_LS}", flush=True)

    ## -- data
    _, val_loader, _, val_sampler = data_loader.build_data_loader(args)
    utils.print_safe(f"Data loaded: val: {len(val_loader)}", flush=True)
    
    ## -- create model
    if args.category_16:
        num_classes = 16
    elif args.category_1k:
        num_classes = 1000
    else:
        num_classes = 209
    classifier = models.get_classifier(args.arch, num_classes=num_classes, pretrained=False)

    if args.append_layer == "bandpass":
        model = models.BandPassNet(classifier, kernel_size=args.kernel_size)
    elif args.append_layer == "blur":
        raise NotImplementedError
        # model = models.BlurNet(classifier)
    else:
        model = classifier
    
    model = models.AttackNet(model, args.append_layer)
    model = model.to(device).eval()
    utils.print_safe(model)

    results = []
    for i, epsilon in enumerate(EPSILON_LS):
        print(f"\n-> Current Epsilon [{i}]/[{len(EPSILON_LS)}]: {epsilon:4f}")

        original_acc_sum, perturb_accs, n = attack_alg.foolbox_attack(val_loader, model, device,
                                                                      epsilon, args.lp, args.attack_alg)

        results.append([original_acc_sum / n * 100, perturb_accs / n * 100])

    for i in range(len(results)):
        print(
            f"Linf norm ≤ {EPSILON_LS[i]:.4f}, "
            f"clean accuracy:  {results[i][0]:.2f}, "
            f"perturbed accuracy: {results[i][1]:.2f}"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.category_16 or args.category_1k:
        args.category_209 = False
    
    args.distributed = False
    args.train_workers = args.workers
    args.test_workers = args.workers

    utils.show_input_args(args)

    main()