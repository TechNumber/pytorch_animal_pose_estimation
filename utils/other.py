import torch


def hmap_to_keypoints(hmap, form='relative'):
    coords = (hmap == torch.amax(hmap, dim=(-1, -2), keepdim=True)).nonzero()[:, -2:].reshape(*hmap.shape[:-2], 2)
    if form == 'relative':
        return coords / torch.ByteTensor(data=hmap.shape[-2:])
    elif form == 'absolute':
        return coords
    else:
        raise NotImplementedError
