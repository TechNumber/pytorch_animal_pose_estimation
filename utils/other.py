import torch


def hmap_to_keypoints(hmap, relative=True):
    # print('orig shape', hmap.shape)
    # print('amax', torch.amax(hmap, dim=(-1, -2), keepdim=True).shape)
    # print('equal', (hmap == torch.amax(hmap, dim=(-1, -2), keepdim=True)).shape)
    # print('nonzero', (hmap.abs() == torch.amax(hmap.abs(), dim=(-1, -2), keepdim=True)).nonzero().shape)
    # print('new shape', (hmap == torch.amax(hmap, dim=(-1, -2), keepdim=True)).nonzero()[:, -2:].reshape(*hmap.shape[:-2], 2).shape)
    # coords = (hmap.abs() == torch.amax(hmap.abs(), dim=(-1, -2), keepdim=True)).nonzero()[:, -2:].reshape(*hmap.shape[:-2], 2)
    h, w = hmap.shape[-2:]
    coords = hmap.abs().flatten(start_dim=-2).max(dim=-1, keepdim=True).indices
    coords = torch.cat([coords // h, coords % h], dim=-1)
    if relative:
        return coords / torch.tensor(data=hmap.shape[-2:], device=hmap.device, dtype=torch.int16)
    else:
        return coords
