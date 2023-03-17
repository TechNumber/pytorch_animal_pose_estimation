import PIL.Image
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
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


def show_plt_keypoints(image, keypoints, show_edges=False, ax=None, as_fig=False):
    assert isinstance(image, (torch.Tensor, PIL.Image.Image))

    if as_fig:
        fig, ax = plt.subplots(figsize=(2, 2))
    elif not ax:
        ax = plt.gca()

    if isinstance(image, torch.Tensor):
        width = image.shape[-2]
        height = image.shape[-3]
    else:
        width = image.width
        height = image.height

    ax.imshow(image)

    if show_edges:
        # Works only with default keypoints layout (16 points, default names from labeling app entries)
        edges = [
            [0, 1, 'lightgreen'],
            [1, 2, 'lightblue'],
            [2, 3, 'lightblue'],
            [3, 4, 'lightblue'],
            [1, 5, 'green'],
            [5, 6, 'green'],
            [6, 7, 'green'],
            [1, 8, 'yellow'],
            [8, 9, 'yellow'],
            [9, 10, 'violet'],
            [10, 11, 'violet'],
            [11, 12, 'violet'],
            [9, 13, 'orange'],
            [13, 14, 'orange'],
            [14, 15, 'orange']
        ]
        for edge in edges:
            ax.plot(
                (keypoints[edge[0], 0] * width, keypoints[edge[1], 0] * width),
                (keypoints[edge[0], 1] * height, keypoints[edge[1], 1] * height),
                linewidth=2,
                c=edge[2]
            )
    else:
        cmap = np.array(['orange', 'green'])
        ax.scatter(
            keypoints[:, 0] * width,
            keypoints[:, 1] * height,
            s=30,
            # marker='.',
            edgecolors='black',
            c=cmap[(keypoints[:, 2] >= 0.5).int()]
        )

    return fig if as_fig else ax


def show_hmaps(hmaps, img=None, kp_names=None):
    if img is not None:
        if isinstance(img, torch.Tensor):
            width = img.shape[-2]
            height = img.shape[-3]
        else:
            width = img.width
            height = img.height
        resize = transforms.Resize((width, height))

    n_kp = hmaps.shape[-3]
    figs_list = []

    for i in range(n_kp):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.axis('off')
        if kp_names:
            ax.set_title(kp_names[i])
        if img is not None:
            ax.imshow(img)
        hmap = resize(hmaps[i].unsqueeze(0)).squeeze() if img is not None else hmaps[i]
        ax.imshow(hmap, alpha=(hmap - hmap.min()) / (hmap.max() - hmap.min()))
        figs_list.append(fig)

    return figs_list
