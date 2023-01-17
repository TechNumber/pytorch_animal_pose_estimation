import PIL.Image
import numpy as np
import torch
from matplotlib import pyplot as plt


def show_keypoints(image, keypoints, show_edges=False):
    assert isinstance(image, (torch.Tensor, PIL.Image.Image))
    if isinstance(image, torch.Tensor):
        width = image.shape[-2]
        height = image.shape[-3]
    else:
        width = image.width
        height = image.height

    plt.imshow(image)
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
            plt.plot((keypoints[edge[0], 0] * width, keypoints[edge[1], 0] * width),
                     (keypoints[edge[0], 1] * height, keypoints[edge[1], 1] * height),
                     linewidth=2,
                     c=edge[2])
    else:
        cmap = np.array(['orange', 'green'])
        sc = plt.scatter(keypoints[:, 0] * width,
                    keypoints[:, 1] * height,
                    s=30,
                    # marker='.',
                    edgecolors='black',
                    c=cmap[(keypoints[:, 2] >= 0.5).int()])
