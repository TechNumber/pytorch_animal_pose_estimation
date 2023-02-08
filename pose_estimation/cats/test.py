import numpy as np
import torch

import wandb
from visualization.keypoints import show_keypoints


def test(model,
         loss,
         data_test,
         logger=None,
         device=torch.device('cpu')):
    loss_value_list = []
    model.eval()
    with torch.inference_mode():
        for batch, batch_data in enumerate(data_test):
            test_img = batch_data['image'].to(device)
            test_hmap = batch_data['heatmap'].to(device)

            pred_hmaps = model(test_img)

            loss_value = loss(pred_hmaps, test_hmap.unsqueeze(1)).item()
            loss_value_list.append(loss_value)

            if logger and batch == 0:
                

        if logger:
            for i in range()
                true_img_ax = show_keypoints(sample['image'], sample['keypoints'], True)
            pred_img =
            table = wandb.Table(columns=["true", "pred"] + [f"score_{i}" for i in range(10)])
            for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
                table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
            wandb.log({"predictions_table": table}, commit=False)

    avg_loss_value = np.average(loss_value_list)

    return avg_loss_value
