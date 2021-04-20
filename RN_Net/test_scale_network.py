import torch
from torch.utils.data import DataLoader
import PIL_Structured3D_Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from utils import angular_error
import os
import glob
import cv2 as cv
from torchvision import transforms
from Real_Dataset import Real_Dataset


def test_model_show_results(testnet, testloader, device, i, loss_nor, loss_alb):
    with torch.no_grad():
        data = testloader.__getitem__(i)
        rgbd_input = data['rgbd'].to(device)
        gt_normal = data['normal'].to(device)
        gt_albedo = data['albedo'].to(device)

        output_nor, output_alb = testnet(rgbd_input)
        print('Normal loss: %.5f   Albedo loss: %.5f' % (loss_nor[i], loss_alb[i]))

        gt_n = gt_normal.to('cpu').numpy()
        # plt.matshow(np.moveaxis(gt_n[0], 0, -1))
        output_n = output_nor.to('cpu').numpy()
        # plt.matshow(np.moveaxis(output_n[0], 0, -1))
        ang_err = angular_error(gt_n[0], output_n[0])
        # plt.matshow(ang_err)

        gt_a = gt_albedo.to('cpu').numpy()
        # plt.matshow(np.round(np.moveaxis(gt_a[0], 0, -1) * 255).astype(np.uint8))
        output_a = output_alb.to('cpu').numpy()
        # plt.matshow(np.round(np.moveaxis(output_a[0], 0, -1) * 255).astype(np.uint8))
    return


def test_model(testnet, dataset_loader, device, data_path):
    loss_func = nn.MSELoss()
    loss_total, loss_nor, loss_alb = np.empty((len(dataset_loader),)), np.empty((len(dataset_loader),)), np.empty(
        (len(dataset_loader),))

    scene_path_list = sorted(glob.glob(os.path.join(data_path, '*')))

    with torch.no_grad():
        for i, data in enumerate(dataset_loader, 0):
            if i < 10:
                small_data = data['small']
                small_rgbd_input = torch.cat([small_data['rgb'].to(device), small_data['depth'].to(device)], dim=1)
                # small_gt_normal = small_data['normal'].to(device)
                # small_gt_albedo = small_data['albedo'].to(device)

                large_data = data['large']
                large_rgbd_input = torch.cat([large_data['rgb'].to(device), large_data['depth'].to(device)], dim=1)

                output = testnet(small_rgbd_input, large_rgbd_input)  # out_nor, out_alb, small_nor, small_alb

                output_nor, output_alb = output[0], output[1]

                output_a = np.moveaxis(output_alb[0].to('cpu').numpy(), 0, -1)
                # output_a = output_a/output_a.max()
                output_n = np.moveaxis(output_nor[0].to('cpu').numpy(), 0, -1)

                if 'normal' in large_data.keys():
                    large_gt_normal = large_data['normal'].to(device)
                    large_gt_albedo = large_data['albedo'].to(device)
                    loss_nor[i] = loss_func(output_nor, large_gt_normal).item()
                    loss_alb[i] = loss_func(output_alb, large_gt_albedo).item()
                    loss_total[i] = loss_nor[i] + loss_alb[i]
                    # gt_n = large_gt_normal.to('cpu').numpy()
                    # ang_err = np.round(angular_error(gt_n[0], output_n[0])).astype(np.uint8)
                    # ang_err = np.expand_dims(angular_error(gt_n[0], output_n[0])/180, 0)

                # writer.add_image("Out Albedo", output_alb[0], i)
                # writer.add_image("Out Normal", output_nor[0], i)
                # writer.add_image("Out Angular error", ang_err, i)

                w_p = os.path.dirname(model_path)
                cv.imwrite(w_p + '/albedo_%05d.png' % i, np.round(((np.clip(output_a, -1, 1) + 1) / 2) * 255).astype(np.uint8)[:, :, ::-1])  # cv write in BGR
                np.save(w_p + '/albedo_%05d.npy' % i, (np.clip(output_a, -1, 1) + 1) / 2)  # save npy in RGB
                cv.imwrite(w_p + '/normal_%05d.png' % i, np.round((output_n + 1) / 2 * 255).astype(np.uint8)[:, :, ::-1])  # cv write in BGR
                np.save(w_p + '/normal_%05d.npy' % i, output_n)  # save npy in xyz

                scene_path = scene_path_list[i]
                cv.imwrite(scene_path + '/output_albedo_coarse.png', np.round(((np.clip(output_a, -1, 1) + 1) / 2) * 255).astype(np.uint8)[:, :, ::-1])  # cv write in BGR
                np.save(scene_path + '/output_albedo_coarse.npy', (np.clip(output_a, -1, 1) + 1) / 2)  # save npy in RGB
                cv.imwrite(scene_path + '/output_normal.png', np.round((output_n + 1) / 2 * 255).astype(np.uint8)[:, :, ::-1])  # cv write in BGR
                np.save(scene_path + '/output_normal.npy', output_n)  # save npy in xyz

                # cv.imwrite(w_p + '/normalerror_%05d.png' % i, ang_err)
            else:
                break

    # print('Total loss: %.5f    Nor loss: %.5f    Albedo loss:%.5f '
    #       % (loss_total.sum()/10, loss_nor.sum()/10, loss_alb.sum()/10))

    # test_model_show_results(testnet, testloader, device, 0, loss_nor, loss_alb)

    return loss_nor, loss_alb


if __name__ == '__main__':
    from ResNet_models import Scale_Network as model_def
    model_path = './RN_Net/trained_model/net_params.pth'
    testnet = model_def()
    testnet.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testnet.to(device)
    testnet.eval()

    #############  Uncomment below to test on the Blender synthetic dataset  ###################
    data_path = './data/synthetic'
    real_data = Real_Dataset(data_path, transformer=PIL_Structured3D_Dataset.AddSmallScaleData(scale=4))
    data_loader = DataLoader(real_data, batch_size=1, shuffle=False, num_workers=0)
    loss_nor, loss_alb = test_model(testnet, data_loader, device, data_path)
    ######################################################################################

    #############  Uncomment below to test on the real dataset  ###################
    data_path = './data/real'
    real_data = Real_Dataset(data_path, transformer=PIL_Structured3D_Dataset.AddSmallScaleData(scale=4))
    data_loader = DataLoader(real_data, batch_size=1, shuffle=False, num_workers=0)
    loss_nor, loss_alb = test_model(testnet, data_loader, device, data_path)
    ######################################################################################
