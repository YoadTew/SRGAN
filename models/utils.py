import torch
from PIL import Image
import torch.nn.functional as F


def load_image(filename, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def ft_map_to_patches(input , patch_size=8):
    B, C, W, H = input.size()

    x = F.unfold(input, kernel_size=8, stride=8)
    x = x.view(B, C, patch_size, patch_size, -1)
    x = x.permute(0, 4, 1, 2, 3)

    return x

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def patch_gram_matrix(y):
    (b, patch, ch, h, w) = y.size()
    features = y.reshape(b*patch, ch, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # batch = batch / 255.0
    return (batch - mean) / std