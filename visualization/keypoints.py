from matplotlib import pyplot as plt


def show_keypoints(image, keypoints, show_edges=False):
    plt.figure()
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
            plt.plot((keypoints[edge[0], 0] * image.width, keypoints[edge[1], 0] * image.width),
                     (keypoints[edge[0], 1] * image.height, keypoints[edge[1], 1] * image.height),
                     linewidth=2,
                     c=edge[2])
    else:
        plt.scatter(keypoints[:, 0] * image.width,
                    keypoints[:, 1] * image.height,
                    s=30,
                    # marker='.',
                    edgecolors='black',
                    c='orange')
