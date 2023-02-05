import numpy as np
import torch


def test(model,
         loss,
         data_test,
         device=torch.device('cpu')):
    loss_value_list = []

    for batch_data in data_test:
        test_img = batch_data['image'].to(device)
        test_hmap = batch_data['heatmap'].to(device)

        pred_hmaps = model(test_img)

        loss_value = loss(pred_hmaps, test_hmap.unsqueeze(1)).item()
        loss_value_list.append(loss_value)

    avg_loss_value = np.average(loss_value_list)
    return avg_loss_value
