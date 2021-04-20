import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from torchvision import transforms
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from shutil import copyfile


class Albedo_Shading_Joint_Opt_Model(nn.Module):
    def __init__(self, albedo=None, shading=None, hy_lamdas=None):
        """
        albedo, shading, image  are in shape: (channel, height, width), float32
        """
        super(Albedo_Shading_Joint_Opt_Model, self).__init__()
        if albedo is None:
            self.albedo = nn.Parameter(torch.zeros(size=(3, 512, 1024), dtype=torch.float32))
        else:
            self.albedo = nn.Parameter(albedo)
        if shading is None:
            self.shading = nn.Parameter(torch.zeros(size=(3, 512, 1024), dtype=torch.float32))
        else:
            self.shading = nn.Parameter(shading)

        self.hy_lamdas = hy_lamdas

    def forward(self, x):
        albedo_input, shading_input, image_input = x

        albedo_pred = self.albedo + albedo_input
        grad_albedo_h = albedo_pred[:,:,0:-2] - albedo_pred[:,:,2:]
        grad_albedo_v = albedo_pred[:, 0:-2, :] - albedo_pred[:, 2:, :]
        l1_norm = (torch.mean(torch.abs(grad_albedo_h)) + torch.mean(torch.abs(grad_albedo_v)))/2

        shading_pred = self.shading + shading_input
        grad_shading_h = shading_pred[:,:,0:-2] - shading_pred[:,:,2:]
        grad_shading_v = shading_pred[:, 0:-2, :] - shading_pred[:, 2:, :]
        l2_norm = (torch.mean(grad_shading_h**2) + torch.mean(grad_shading_v**2))/2

        pred_img = albedo_pred * shading_pred
        scale = scale_compute(image_input, pred_img)
        reconst = F.mse_loss(image_input, pred_img * scale)

        weight_decay_alb = torch.mean(self.albedo ** 2)
        weight_decay_sha = torch.mean(self.shading ** 2)

        energy = self.hy_lamdas[0]*reconst + self.hy_lamdas[1]*l1_norm + self.hy_lamdas[2]*l2_norm + \
                 self.hy_lamdas[3]*weight_decay_alb + self.hy_lamdas[4]*weight_decay_sha
        return energy, reconst


def scale_compute(gt, prediction):
    scale, _ = torch.lstsq(gt.flatten().unsqueeze(1), prediction.flatten().unsqueeze(1))
    return scale[0, 0].clone().detach()


if __name__ == '__main__':
    # it runs faster in CPU
    device = torch.device("cpu")

    scene_name_list = sorted(glob.glob('./data/*/*'))
    for path in scene_name_list:
        scene_name = os.path.basename(path)

        model = Albedo_Shading_Joint_Opt_Model(hy_lamdas=[10, 1, 100, 1, 10])
        model.to(device)

        log_path = './TV_optimization/runs/'+scene_name
        writer = SummaryWriter(log_path)

        albedo = cv.imread(os.path.join(path, 'output_albedo_coarse.png'))[:,:,::-1].astype(np.float32)/255
        image = cv.imread(os.path.join(path, 'up.png'))[:,:,::-1].astype(np.float32)/255
        # load and tonemap shading
        shading = np.load(os.path.join(path, 'shading_full.npy')).astype(np.float32)
        shading = shading**(1/2.2)
        shading = np.mean(shading, axis=-1, keepdims=True)
        if shading.max() > 1 or shading.max() < 0.3:
            shading = shading / shading.max()
        albedo = transforms.ToTensor()(albedo)
        shading = transforms.ToTensor()(shading)
        image = transforms.ToTensor()(image)

        # load albedo gt
        # albedo_gt = cv.imread(os.path.join(path, 'albedo_gt.png'))[:,:,::-1].astype(np.float32)/255
        # albedo_gt_mask = np.sum(albedo_gt, axis=-1)
        # albedo_gt_mask[np.where(albedo_gt_mask >= 1e-3)] = 1
        # albedo_gt_mask[np.where(albedo_gt_mask < 1e-3)] = 0
        # albedo_gt = transforms.ToTensor()(albedo_gt)
        # albedo_gt_mask = torch.unsqueeze(torch.from_numpy(albedo_gt_mask), dim=0)
        # scale = scale_compute(albedo_gt * albedo_gt_mask, albedo * albedo_gt_mask)
        # accuratcy = torch.sum((albedo_gt * albedo_gt_mask - scale * albedo * albedo_gt_mask) ** 2) / torch.sum(albedo_gt_mask)
        # writer.add_scalar('Abledo accuratcy', accuratcy.item(), 0)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)

        start_t = time.time()
        num_iter = 1000
        running_loss_total = 0

        for i in range(num_iter):
            optimizer.zero_grad()  # zero the gradient buffers
            albedo, shading, image = albedo.to(device), shading.to(device), image.to(device)
            loss, reconst_loss = model([albedo, shading, image])
            loss.backward()
            optimizer.step()

            running_loss_total += loss.item()
            writer.add_scalar('Total loss', loss.item(), i)
            writer.add_scalar('Reconst loss', reconst_loss.item(), i)
            #
            # scale = scale_compute(albedo_gt*albedo_gt_mask, (albedo+model.albedo)*albedo_gt_mask)
            # accuratcy = torch.sum((albedo_gt*albedo_gt_mask - scale * (albedo+model.albedo)*albedo_gt_mask)**2) / torch.sum(albedo_gt_mask)
            # writer.add_scalar('Abledo accuratcy', accuratcy.item(), i+1)
            scheduler.step()

            if i % 100 == 0:
                cost_t = time.time() - start_t
                est_time = cost_t / (i + 1) * (num_iter - i - 1)
                print('Iter: %5d/%5d,  loss: %.3f,  cost_time: %d m %2d s,  est_time: %d m %2d s' %
                      (i + 1, num_iter, running_loss_total / 100, cost_t // 60, cost_t % 60,
                       est_time // 60, est_time % 60))
                running_loss_total = 0

                # create grid of images
                alb_c = (model.albedo+albedo).detach()
                img_grid = torchvision.utils.make_grid(alb_c/alb_c.max())
                writer.add_image('Albedo ', img_grid, global_step=i)

                s_c = (model.shading+shading).detach()
                img_grid = torchvision.utils.make_grid(s_c/s_c.max())
                writer.add_image('Shading ', img_grid, global_step=i)

                pred_img = alb_c * s_c
                img_grid = torchvision.utils.make_grid(pred_img/pred_img.max())
                writer.add_image('Reconstruct ', img_grid, global_step=i)

                gt_img_c = image
                scale = scale_compute(gt_img_c, pred_img)
                img_grid = torchvision.utils.make_grid(torch.abs(image - scale*pred_img))
                writer.add_image('Abs Difference ', img_grid, global_step=i)

        torch.save(model.state_dict(), log_path + '/net_params.pth')
        copyfile(__file__, log_path + '/trained_setting.py')
        refine_albedo = (albedo + model.albedo).detach().cpu().numpy()
        refine_albedo = refine_albedo / refine_albedo.max()
        refine_albedo = np.round(np.moveaxis(refine_albedo, 0, -1) * 255).astype(np.uint8)
        cv.imwrite(log_path + '/output_albedo_refined.png', refine_albedo[:,:,::-1])
        cv.imwrite(path + '/output_albedo_refined.png', refine_albedo[:, :, ::-1])
