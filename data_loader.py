
import os
from PIL import Image
import numpy as np

import torch
from torchvision import transforms
import h5py
import utils


############## HYPERPARAMETERS ##############

T_ImageNet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


CATEGORIES16 = ["airplane", 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair', 'clock', 'dog', 
	       'elephant', 'keyboard', 'knife', 'oven', 'truck']
# these were wordnet categories, so not all were included in imagenet, manually removed those
CAT2SNET = {
    "knife" :    ["n03041632"],
    "keyboard" : ["n03085013", "n04505470"],
    "elephant" : ["n02504013", "n02504458"],
    "bicycle" :  ["n02835271", "n03792782"],
    "airplane" : ["n02690373", "n03954731"], # , "n13861050", "n13941806"], # "n03955296"
    "clock" :    ["n02708093", "n03196217", "n04548280"],
    "oven" :     ["n03259280", "n04111531"], # "n03259401", "n04111414", 
    "chair" :    ["n02791124", "n03376595", "n04099969", "n04429376"], # "n00605023", 
    "bear" :     ["n02132136", "n02133161", "n02134084", "n02134418"],
    "boat" :     ["n02951358", "n03344393", "n03662601", "n04273569", "n04612504"], # "n04612373", 
    "cat" :      ["n02123045", "n02123159", "n02123394", "n02123597", "n02124075", 
                  "n02125311"], # "n02122878", "n02126465",
    "bottle" :   ["n02823428", "n03937543", "n03983396", "n04557648", "n04560804", 
                  "n04579145", "n04591713"],
    "truck" :    ["n03345487", "n03417042", "n03770679", "n03796401", "n03930630", 
                  "n03977966", "n04461696", "n04467665"],  # "n00319176", "n01016201", "n03930777", n05061003, n06547832, n10432053
    "car" :      ["n02814533", "n03100240", "n04285008"], # n03100346, n13419325
    "bird" :     ["n01514859", "n01530575", "n01531178", 
                  "n01532829", "n01534433", "n01537544", "n01558993",
                  "n01560419", "n01582220", "n01592084", "n01601694", 
                  "n01614925", "n01616318", "n01622779", "n01795545", "n01796340", 
                  "n01797886", "n01798484", "n01817953", "n01818515", "n01819313", 
                  "n01820546", "n01824575", "n01828970", "n01829413", "n01833805", 
                  "n01843065", "n01843383", "n01855032", "n01855672",
                  "n01860187", "n02002556", "n02002724", "n02006656", "n02007558", 
                  "n02009229", "n02009912", "n02011460", "n02013706", "n02017213", 
                  "n02018207", "n02018795", "n02025239", "n02027492", "n02028035", 
                  "n02033041", "n02037110", "n02051845", "n02056570"], 
                  # n01321123, n01792640, n07646067 n01562265, n10281276 n07646821
    "dog" :      ["n02085782", "n02085936", "n02086079", "n02086240", "n02086646", 
                  "n02086910", "n02087046", "n02087394", "n02088094", "n02088238", 
                  "n02088364", "n02088466", "n02088632", "n02089078", "n02089867", 
                  "n02089973", "n02090379", "n02090622", "n02090721", "n02091032", 
                  "n02091134", "n02091244", "n02091467", "n02091635", "n02091831", 
                  "n02092002", "n02092339", "n02093256", "n02093428", "n02093647", 
                  "n02093754", "n02093859", "n02093991", "n02094114", "n02094258", 
                  "n02094433", "n02095314", "n02095570", "n02095889", "n02096051", 
                  "n02096294", "n02096437", "n02096585", "n02097047", "n02097130", 
                  "n02097209", "n02097298", "n02097474", "n02097658", "n02098105", 
                  "n02098286", "n02099267", "n02099429", "n02099601", "n02099712", 
                  "n02099849", "n02100236", "n02100583", "n02100735", "n02100877", 
                  "n02101006", "n02101388", "n02101556", "n02102040", "n02102177", 
                  "n02102318", "n02102480", "n02102973", "n02104029", "n02104365", 
                  "n02105056", "n02105162", "n02105251", "n02105505", "n02105641", 
                  "n02105855", "n02106030", "n02106166", "n02106382", "n02106550", 
                  "n02106662", "n02107142", "n02107312", "n02107574", "n02107683", 
                  "n02107908", "n02108000", "n02108422", "n02108551", "n02108915", 
                  "n02109047", "n02109525", "n02109961", "n02110063", "n02110185",
                  "n02110627", "n02110806", "n02110958", "n02111129", "n02111277", 
                  "n02111500", "n02112018", "n02112350", "n02112706", 
                  "n02113023", "n02113624", "n02113712", "n02113799", "n02113978"], 
                # n08825211
}


############## DATA LOADER ##############

def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class DatasetLdrImageNet:
    transform_ = [transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor()]

    def __init__(self, folder_root, folder_ls_txt, 
                 orig_imagenet_mapping=None, use_imagenet_transform=True,
                 subset_ldr=False, seed = 415):
        self.root = folder_root
        self.folder_ls, self.class_to_name = self._load_img_folder_ls(folder_ls_txt)

        self.classes, self.class_to_idx = self.find_classes()  # e.g. "'n01440764': 0"
        self.classname2idx = {self.class_to_name[k]: self.class_to_idx[k] for k in self.class_to_name}

        if orig_imagenet_mapping:
            self.classname2origidx = self._get_orig_mapping(orig_imagenet_mapping)

        self.min_n, self.samples = self.make_dataset(subset_ldr, seed)

        self.loader = pil_loader
        self.targets = [s[1] for s in self.samples]

        if use_imagenet_transform:
            self.transform = transforms.Compose(self.transform_ + [T_ImageNet])
        else:
            self.transform = transforms.Compose(self.transform_ + [T_NSDCOCO])

    def _get_orig_mapping(self, orig_imagenet_mapping):
        mapping = utils.pickle_load(orig_imagenet_mapping)
        return {self.class_to_name[k]: mapping[k] for k in self.class_to_name}

    def _load_img_folder_ls(self, txt_pth):
        cls2name = {}
        with open(txt_pth, 'r') as f:
            folder_ls = []
            for ln in f.readlines():
                cls, name = ln.split(':')

                folder_ls.append(cls.strip())
                cls2name[cls.strip()] = name.strip()
            
            return folder_ls, cls2name

    def find_classes(self):
        classes = sorted(self.folder_ls)  # !! alphabetical!
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def make_dataset(self, subset_ldr, seed):
        min_n = 1e10
        if subset_ldr:
            for f in self.folder_ls:
                img_folder_pth = os.path.join(self.root, f)
                curr_n = 0
                for img_f in sorted(os.listdir(img_folder_pth)):
                    if img_f.endswith(".JPEG") or img_f.endswith(".jpeg"):
                        curr_n += 1
                min_n = min(min_n, curr_n)

        samples = []
        for f in self.folder_ls:
            img_folder_pth = os.path.join(self.root, f)

            curr_sample_ls = []
            for img_f in sorted(os.listdir(img_folder_pth)):
                if img_f.endswith(".JPEG") or img_f.endswith(".jpeg"):
                    img_f_pth = os.path.join(img_folder_pth, img_f)
                    class_name = self.class_to_name[f]
                    curr_sample_ls.append((img_f_pth, class_name))
            if subset_ldr:
                rng = np.random.default_rng(seed=seed)
                chosen_samples = rng.choice(np.arange(len(curr_sample_ls)), size=min_n, replace=False)
                samples.extend([curr_sample_ls[i] for i in chosen_samples])
            else:
                samples.extend(curr_sample_ls)

        return min_n, samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):

        path, target = self.samples[index]
        img = self.loader(path)
        img = self.transform(img)

        return img, target


def load_data_ImageNet(img_dir, folder_ls_txt, batch_size, num_workers, 
                       orig_imagenet_mapping=None, use_imagenet_transform=True,
                       subset_ldr=False):
    dset = DatasetLdrImageNet(img_dir, folder_ls_txt, orig_imagenet_mapping, use_imagenet_transform, subset_ldr)
    im_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    return im_loader


def load_img_folder_ls(txt_pth):
    with open(txt_pth, 'r') as f:
        folder_ls = []
        for ln in f.readlines():
            folder_ls.append(ln.split(':')[0].strip())
        return folder_ls
    

def build_data_loader(args):
    img_folder_ls = load_img_folder_ls(args.img_folder_txt)

    ### data Loader
    train_loader, val_loader, train_sampler, val_sampler = \
        load_data_folder(args.data, img_folder_ls, args.distributed, args.batch_size,
                                     args.train_workers, args.test_workers, 
                                     shuffle=False,
                                     orig_imagenet_lbs=args.orig_imagenet_lbs)

    
    return train_loader, val_loader, train_sampler, val_sampler








# def load_data_adv(adv_imgs, lbs, batch_size, num_workers):
#     dset = DatasetLdrAdv(adv_imgs, lbs)
#     im_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
#                                             shuffle=False, num_workers=num_workers)

#     return im_loader


# class DatasetLdrAdv:
#     """Used by utils_adv.py"""

#     def __init__(self, adv_imgs, lbs):

#         self.adv_imgs = adv_imgs
#         self.lbs = lbs

#     def __len__(self):
#         return len(self.lbs)

#     def __getitem__(self, idx):
#         image = self.adv_imgs[idx]
#         lb = self.lbs[idx]

#         return image, lb



