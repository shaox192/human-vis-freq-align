from foolbox import PyTorchModel, accuracy
from foolbox.attacks import LinfPGD, LinfFastGradientAttack, L2CarliniWagnerAttack, \
    L2DeepFoolAttack, L2FastGradientAttack, L2BrendelBethgeAttack


ATTACK_ALG = {"linf": {"pgd": LinfPGD,
                       "fgsm": LinfFastGradientAttack},
              "l2": {"fgsm": L2FastGradientAttack,
                     "deepfool": L2DeepFoolAttack}
        }

def foolbox_attack(val_loader, 
                   model, 
                   device,
                   epsilon, 
                   lp, 
                   attack_alg, 
                   kwargs={}
                   ):
    """
    val_loader: dataloader for validation set
    model: model to attack
    device:
    epsilon: perturbation size
    """

    fmodel = PyTorchModel(model, bounds=(-3.0, 3.0))
    try: 
        attack = ATTACK_ALG[lp][attack_alg](*kwargs)
    except KeyError:
        raise "attack type not implemented!"

    original_acc_sum = 0.
    n = 0
    # epsilons = [epsilon]
    # perturb_accs = [0. for _ in epsilons]
    perturb_acc = 0.
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        output = fmodel(images)
        output = fmodel(images).argmax(axis=-1)
        clean_acc = accuracy(fmodel, images, target)

        n += len(images)
        original_acc_sum += clean_acc * len(images)
        # print(f"[{i + 1}]/[{len(val_loader)}]")
        # print(f"clean accuracy:  {clean_acc * 100:.2f}, avg so far: {original_acc_sum / n * 100:.2f}", flush=True)
        # try:
        raw_advs, clipped_advs, success = attack(fmodel, images, target, epsilons=[epsilon])
        # print("------ finished attack -------", flush=True)

        robust_accuracy = 1 - success.float().mean(axis=-1)
        # print("robust accuracy for perturbations with", end=': ')
        # for i, (eps, acc) in enumerate(zip(epsilons, robust_accuracy)):
        perturb_acc += robust_accuracy[0].item() * len(images)
        # print(f"Linf norm ≤ {epsilons[0]:.4f}: {acc.item() * 100:.2f}, avg so far: {perturb_accs[i] / n * 100:.2f}", flush=True)
    
    return original_acc_sum, perturb_acc, n


