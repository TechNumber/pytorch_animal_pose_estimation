import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_pose(image, pose, ignore_joints = None):
    t = transforms.ToPILImage()
    # image -> tensor, pose -> tensor[28]
    image_size = image.shape[1]
    plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(121, aspect='equal')
    plt.imshow(t(image))
    pose = pose.reshape(14, 2)
    for i, j in [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11), (8, 12), (9, 12), (12, 13), (3, 12), (2, 12)]:
        if ignore_joints == None:
            plt.plot([pose[i][0] * image_size, pose[j][0]* image_size], [pose[i][1] * image_size, pose[j][1] * image_size ], '-o',lw=2)
        else: 
            if ignore_joints[i] and ignore_joints[j]:
                plt.plot([pose[i][0] * image_size, pose[j][0]* image_size], [pose[i][1] * image_size, pose[j][1] * image_size ], '-o',lw=2)
    
    plt.subplot(122, aspect='equal', sharex=ax1, sharey=ax1)

    for i, j in [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11), (8, 12), (9, 12), (12, 13), (3, 12), (2, 12)]:
        if ignore_joints == None:
            plt.plot([pose[i][0] * image_size, pose[j][0]* image_size], [pose[i][1] * image_size, pose[j][1] * image_size ], '-o',lw=2)
        else: 
            if ignore_joints[i] and ignore_joints[j]:
                plt.plot([pose[i][0] * image_size, pose[j][0]* image_size], [pose[i][1] * image_size, pose[j][1] * image_size ], '-o',lw=2)

def show_pose_from_hmap(image, hmap, mask = None):
    pose = points_from_hmap(hmap)
    show_pose(image, pose, mask)

def points_from_hmap(hmaps):
    points = []
    for hmap in hmaps:
        x, y = np.unravel_index(np.argmax(hmap), hmap.shape)
        points.append(y/hmap.shape[1])
        points.append(x/hmap.shape[0])
    return np.array(points)

def show_hmap(image, hmap):
    t = transforms.ToPILImage()
    # t2 = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor()
    # ])
    
    plt.figure(figsize=(20, 30))
    plt.subplot(6, 3, 1)
    plt.imshow(t(image))
    
    for i, h in enumerate(hmap):
        plt.subplot(6, 3, i + 2)
        plt.imshow(h)
    
    # plt.figure(figsize=(20, 30))
    # plt.subplot(6, 3, 1)
    # plt.imshow(t(torch.cat((image, t2(c_hmap)))))
    # for j, h in enumerate(hmap):
    #     plt.subplot(6, 3, j + 2)
    #     plt.imshow(t(torch.cat((image, t2(h)))))

def show_hmap_cpm(image, hmap, joint):
    t = transforms.ToPILImage()

    plt.figure(figsize=(20, 30))
    plt.subplot(1, 7, 1)
    plt.imshow(t(image))
    for i, prediction in enumerate(hmap):
        plt.subplot(1, 7, i + 2)
        plt.imshow(prediction.squeeze()[joint].squeeze().cpu().detach())