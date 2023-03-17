import torch


class PCK(torch.nn.Module):
    def __init__(self, thr=0.2):
        super(PCK, self).__init__()
        self.thr = thr

    def forward(self, kp_pred, kp_true):
        max_xy = kp_true.max(dim=-2).values
        min_xy = kp_true.min(dim=-2).values
        box_diag = ((max_xy - min_xy) ** 2).sum(dim=-1, keepdim=True).sqrt()
        true_pred_dist = ((kp_pred - kp_true) ** 2).sum(dim=-1).sqrt()
        return (true_pred_dist < box_diag * self.thr).sum() / true_pred_dist.numel()
