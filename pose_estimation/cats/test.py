import numpy as np
import torch
import wandb

from visualization.keypoints import show_keypoints, show_hmaps


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

        if logger:
            table = wandb.Table(
                columns=["true_keypoints", "pred_keypoints"] + \
                        [f"{k}_probability_heatmap" for k in data_test.dataset.keypoint_names]
            )
            # img, kp = data_test.dataset[0]['image'].movedim(0, -1).cpu(), data_test.dataset[0]['keypoints']
            sample = next(iter(data_test))
            img, kp = sample['image'][0].to(device).unsqueeze(0), sample['keypoints'][0].to(device)

            pred_hmaps = model(img)
            pred_hmaps = pred_hmaps[-1][-1].movedim(-2, -1).detach().cpu()
            pred_kp = (pred_hmaps == torch.amax(
                pred_hmaps, dim=(-2, -1), keepdim=True
            )).nonzero()[:, 1:].numpy() / (pred_hmaps.shape[-2], pred_hmaps.shape[-1])

            img = img[0].movedim(0, -1).cpu()
            true_kp_img = show_keypoints(img, kp[-1].cpu(), show_edges=True, as_fig=True)
            pred_kp_img = show_keypoints(img, pred_kp, show_edges=True, as_fig=True)
            pred_hmaps_imgs = show_hmaps(pred_hmaps, img, data_test.dataset.keypoint_names)

            table.add_data(wandb.Image(true_kp_img), wandb.Image(pred_kp_img), *map(wandb.Image, pred_hmaps_imgs))
            # for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
            #     table.add_data(wandb.Image(img[0].numpy() * 255), pred, targ, *prob.numpy())
            wandb.log({"predictions_table": table}, commit=True)

    avg_loss_value = np.average(loss_value_list)

    return avg_loss_value
