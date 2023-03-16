import torch
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_keypoints, make_grid


def show_keypoints(kp, img, as_grid=True):
    kp = (kp.unsqueeze(-3) * torch.tensor(data=img.shape[-2:], device=img.device, dtype=torch.int16)).flip(dims=[-1])
    img = convert_image_dtype(img, dtype=torch.uint8)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8],
        [8, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15]
    ]
    img = [draw_keypoints(img[i], kp[i], connectivity=edges, colors="blue", radius=5) for i in range(len(img))]
    return [make_grid(img, nrow=4) / 255] if as_grid else img
