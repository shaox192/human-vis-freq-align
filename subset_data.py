
"""
Extract the 16 general categories from the imagenet 1k categories
save to txt file
"""


# import h5py
import numpy as np
import data_loader

# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset
# import torchvision.models as models

# from PIL import Image
# from utils import pickle_load, pickle_dump

# from torch.utils.data import DataLoader



# class ImgSet(Dataset):
#     # mean, std calculated from the subset of training images in intact/scrambled task
#     TRAIN_MEAN = [0.485, 0.456, 0.406]
#     TRAIN_STD = [0.229, 0.224, 0.225]

#     IMG_TF = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
#     ])

#     def __init__(self, input_img_data):
#         self.img_data = input_img_data

#     def __len__(self):
#         return self.img_data.shape[0]

#     def __getitem__(self, idx):
#         orig_img = self.img_data[idx].transpose(1,2,0)
#         tf_img = self.IMG_TF(orig_img)
#         return orig_img, tf_img


# def test(model, data_ldr, category_meta):

#     category_names = []
#     conf_score = []
#     for i, (orig_img, tf_img) in enumerate(data_ldr):

#         # orig_img, tf_img = orig_img.to(DEVICE), tf_img.to(DEVICE)

#         prediction = model(tf_img).squeeze(0).softmax(0)
#         class_id = prediction.argmax().item()

#         score = prediction[class_id].item()
#         conf_score.append(score)

#         category = category_meta["categories"][class_id]
#         category_names.append(category)

#         if i % 20 == 0:
#             print(f"[{i}]/[{len(data_ldr)}]: {len(category_names)} images processed, "
#                   f"last one: {category_names[-1]}, {conf_score[-1]:.4f}")
#     return category_names, conf_score


# def img_show(img_arr, title):
#     if type(img_arr) is torch.Tensor:
#         img_arr = img_arr.squeeze(0)
#         img_arr = img_arr.numpy()

#     im = Image.fromarray(img_arr)
#     im.show(title=title)


# def load_data():
#     img_pth = "stimuli_coco/stim_per_subject/S1_stimuli_227.h5py"
#     image_data_h5py = h5py.File(img_pth, 'r')
#     image_data = np.copy(image_data_h5py['stimuli'])
#     image_data_h5py.close()

#     beta_f = "ExtractNeuralData/nsd/sub1_betas_masked/sub1_orders_betas_VO.pkl"
#     order_data_orig = pickle_load(beta_f)

#     val_img_id = order_data_orig["val_imgID"]
#     train_img_id = order_data_orig["train_imgID"]
#     full_id = np.append(val_img_id, train_img_id)

#     img_norep = image_data[full_id]
#     img_ldr = ImgSet(img_norep)

#     return img_ldr


# def main_get_lb():

#     img_ldr = load_data()
#     data_ldr = DataLoader(img_ldr, batch_size=1, shuffle=False)

#     model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
#     model.eval()

#     category_meta = models.ResNet101_Weights.IMAGENET1K_V2.meta

#     category_names, conf_score = test(model, data_ldr, category_meta)

#     pickle_dump({"category_names": category_names, "conf_score": conf_score}, "coco_imagenet_classes.pkl")


# def find_freq(category_ls):
#     category_dict = {}

#     for i in range(len(category_ls)):
#         c = category_ls[i]
#         if c not in category_dict:
#             category_dict[c] = [i]
#         else:
#             category_dict[c].append(i)
#     return category_dict


def load_imagenet_categories(fname):
    with open(fname, 'r') as f:
        full_ls = f.readlines()

    imgnet_dict = {}
    for category in full_ls:
        category_ls = category.split(' ')
        folder_name = category_ls[0]
        imgnet_dict[folder_name] = []

        txt_names = category_ls[1:] if category_ls[1:] is list else list(category_ls[1:])
        txt_names = ' '.join(txt_names)
        txt_names = txt_names.split(',')
        for t in txt_names:
            t = t.strip()
            imgnet_dict[folder_name].append(t)
    return imgnet_dict


def main():

    save_f = "./data/human16-{}.txt"
    imgnet_1k_f = "./data/imagenet_categories.txt"

    imgnet_1k = load_imagenet_categories(imgnet_1k_f)
    for k, v in imgnet_1k.items():
        print(f"{k}: {v}")

    cat2keep = []
    for k, v in data_loader.CAT2SNET.items():
        for c in v:
            if c not in imgnet_1k:
                print(f"!!!! {c} not in imagenet 1k categories")
            else:
                cat2keep.append((c, imgnet_1k[c][0]))
    print(f"Number of categories (belonged to the 16 cats) to keep {len(cat2keep)}")

    save_f = save_f.format(len(cat2keep))
    with open(save_f, 'w') as f:
        for c, cat_name in cat2keep:
            f.write(f"{c}: {cat_name}\n")


if __name__ == "__main__":
    main()