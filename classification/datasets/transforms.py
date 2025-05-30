import random

import einops
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from scipy import ndimage
from timm.models.layers import to_3tuple


def load_nii_file(nii_image):
    image = sitk.ReadImage(nii_image)
    image_array = sitk.GetArrayFromImage(image)
    return image_array


def resize3D(image, size):
    size = to_3tuple(size)
    image = image.astype(np.float32)
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(image, size=size, mode='trilinear', align_corners=True).squeeze(0).squeeze(0)
    return x.cpu().numpy()


def image_normalization(image, win=None, adaptive=True):
    if win is not None:
        image = np.divide(image - win[0][:, None, None], win[1][:, None, None])
        image[image < 0] = 0.
        image[image > 1] = 1.
        return image
    elif adaptive:
        min, max = np.min(image), np.max(image)
        image = (image - min) / (max - min)
        # image = image - np.mean(image) / np.std(image)])
        return image
    else:
        return image


def random_crop(image, crop_shape):
    crop_shape = to_3tuple(crop_shape)
    _, z_shape, y_shape, x_shape = image.shape

    z_min = np.random.randint(0, z_shape - crop_shape[0])
    y_min = np.random.randint(0, y_shape - crop_shape[1])
    x_min = np.random.randint(0, x_shape - crop_shape[2])
    image = image[..., z_min:z_min + crop_shape[0], y_min:y_min + crop_shape[1], x_min:x_min + crop_shape[2]]
    return image


def center_crop(image, target_shape=(10, 80, 80)):
    target_shape = to_3tuple(target_shape)
    b, z_shape, y_shape, x_shape = image.shape
    z_min = z_shape // 2 - target_shape[0] // 2
    y_min = y_shape // 2 - target_shape[1] // 2
    x_min = x_shape // 2 - target_shape[2] // 2
    image = image[:, z_min:z_min + target_shape[0], y_min:y_min + target_shape[1], x_min:x_min + target_shape[2]]
    return image


def randomflip_z(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, ::-1, ...]


def randomflip_x(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[..., ::-1]


def randomflip_y(image, p=0.5):
    if random.random() > p:
        return image
    else:
        return image[:, :, ::-1, ...]


def random_flip(image, mode='x', p=0.5):
    if mode == 'x':
        image = randomflip_x(image, p=p)
    elif mode == 'y':
        image = randomflip_y(image, p=p)
    elif mode == 'z':
        image = randomflip_z(image, p=p)
    else:
        raise NotImplementedError(f'Unknown flip mode ({mode})')
    return image


def rotate(image, angle=10):
    angle = random.randint(-10, 10)
    r_image = ndimage.rotate(image, angle=angle, axes=(-2, -1), reshape=True)
    if r_image.shape != image.shape:
        r_image = center_crop(r_image, target_shape=image.shape[1:])
    return r_image


def add_noise(image, strength):
    noise = torch.normal(0, strength/255, image.shape).cuda()
    noise_image = torch.clamp(image + noise, 0, 1)
    return noise_image


def add_random_mask(image, ratio, patch_size=7):
    # b, p, c, h, w
    b, p, c, h, w = image.shape
    h_s, w_s = h // patch_size, w // patch_size
    patches = einops.rearrange(image, "b p c (h s1) (w s2) -> b (p c h w) s1 s2", s1=patch_size, s2=patch_size)
    num_mask = int(patches.shape[1] * ratio)
    indices = torch.randint(low=0, high=patches.shape[1], size=(b, num_mask, 1, 1)).cuda()
    indices = indices.repeat([1, 1, patch_size, patch_size])
    patches = patches.scatter(dim=1, index=indices, value=0)
    mask_image = einops.rearrange(patches, "b (p c h w) s1 s2 -> b p c (h s1) (w s2)", p=p, c=c, h=h_s, w=w_s)
    return mask_image



