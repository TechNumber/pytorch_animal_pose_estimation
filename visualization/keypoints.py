import PIL.Image
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


def show_keypoints(image, keypoints, show_edges=False, ax=None):
    assert isinstance(image, (torch.Tensor, PIL.Image.Image))

    if not ax:
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
    return ax


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
        fig, ax = plt.subplots()
        ax.axis('off')
        if kp_names:
            ax.set_title(kp_names[i])
        if img is not None:
            ax.imshow(img)
        hmap = resize(hmaps[i].unsqueeze(0)).squeeze() if img is not None else hmaps[i]
        ax.imshow(hmap, alpha=(hmap - hmap.min()) / (hmap.max() - hmap.min()))
        figs_list.append(fig)

    return figs_list
