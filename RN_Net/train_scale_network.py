import torch
from ResNet_models import Scale_Network
import utils
import torch.optim as optim
from torch.utils.data import DataLoader
import PIL_Structured3D_Dataset
import time
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from torchvision import transforms

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testnet = Scale_Network(scale_factor=4, small_model_pretrained_path=None)

    ##### continue from checkpoint #####
    testnet.load_state_dict(torch.load('./RN_Net/trained_model/net_params.pth'))
    ###################################

    testnet.to(device)
    testnet.train()

    log_path = './RN_Net/runs/exp1'
    writer = SummaryWriter(log_path)

    alb_k = 1
    coarse_k = 0.5
    optimizer = optim.Adam(testnet.parameters(), lr=0.001 * 0.1, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    cus_transform = transforms.Compose([PIL_Structured3D_Dataset.ToTorchTensor(),
                                        PIL_Structured3D_Dataset.NormaliseTensors(),
                                        PIL_Structured3D_Dataset.AddGaussianNoise(std=0.03, scale=1 / 8),
                                        PIL_Structured3D_Dataset.AddGaussianNoise_RGB(std=0.02),
                                        PIL_Structured3D_Dataset.AddSmallScaleData(scale=4)])

    train_dataset = PIL_Structured3D_Dataset.PIL_Structured3D_Dataset('/data/Structured3D_dataset/Structured3D', transformer=cus_transform)
    trainloader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=0)
    test_dataset = PIL_Structured3D_Dataset.PIL_Structured3D_Dataset('/data/Structured3D_dataset/Structured3D', transformer=cus_transform, training=False)
    testloader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=0)

    num_train_data = len(trainloader)

    for epoch in range(0, 50):
        start_t = time.time()
        running_loss_total, running_loss_nor, running_loss_alb = 0, 0, 0
        running_loss_nor_small, running_loss_alb_small = 0, 0
        for i, data in enumerate(trainloader, 0):
            small_data = data['small']
            small_rgbd_input = torch.cat([small_data['rgb'].to(device), small_data['depth'].to(device)], dim=1)
            small_gt_normal = small_data['normal'].to(device)
            small_gt_albedo = small_data['albedo'].to(device)

            large_data = data['large']
            large_rgbd_input = torch.cat([large_data['rgb'].to(device), large_data['depth'].to(device)], dim=1)
            large_gt_normal = large_data['normal'].to(device)
            large_gt_albedo = large_data['albedo'].to(device)

            # in training loop:
            optimizer.zero_grad()  # zero the gradient buffers
            output = testnet(small_rgbd_input, large_rgbd_input)  # out_nor, out_alb, small_nor, small_alb
            gt_list = [small_gt_normal, small_gt_albedo, large_gt_normal, large_gt_albedo]
            loss, loss_nor, loss_alb, small_loss_nor, small_loss_alb = utils.small_large_loss(output, gt_list, alb_k, coarse_k)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss_total += loss.item()
            running_loss_nor += loss_nor.item()
            running_loss_alb += loss_alb.item()
            running_loss_nor_small += small_loss_nor.item()
            running_loss_alb_small += small_loss_alb.item()
            if i % 10 == 9:
                cost_t = time.time() - start_t
                est_time = cost_t/(i+1)*(num_train_data-i-1)
                print('epoch: %d,  iter: %5d/%5d,  loss: %.3f,  cost_time: %d m %2d s,  est_time: %d m %2d s' %
                      (epoch + 1, i + 1, num_train_data, running_loss_total/10, cost_t//60, cost_t % 60, est_time//60, est_time % 60))

                # log the running loss
                writer.add_scalar('Total training loss', running_loss_total / 10, epoch * num_train_data + i)
                writer.add_scalar('Normal training loss', running_loss_nor / 10, epoch * num_train_data + i)
                writer.add_scalar('Albedo training loss', running_loss_alb / 10, epoch * num_train_data + i)
                writer.add_scalar('Normal 1-4 scale training loss', running_loss_nor_small / 10, epoch * num_train_data + i)
                writer.add_scalar('Albedo 1-4 scale training loss', running_loss_alb_small / 10, epoch * num_train_data + i)
                running_loss_total, running_loss_nor, running_loss_alb = 0, 0, 0
                running_loss_nor_small, running_loss_alb_small = 0, 0
        scheduler.step()

        if epoch % 1 == 0 and testloader is not None:  # test model every ? epoch
            testnet.eval()
            with torch.no_grad():
                test_loss_total, test_loss_nor, test_loss_alb, test_ang_err = 0, 0, 0, 0
                for i, data in enumerate(testloader, 0):
                    small_data = data['small']
                    small_rgbd_input = torch.cat([small_data['rgb'].to(device), small_data['depth'].to(device)], dim=1)
                    small_gt_normal = small_data['normal'].to(device)
                    small_gt_albedo = small_data['albedo'].to(device)

                    large_data = data['large']
                    large_rgbd_input = torch.cat([large_data['rgb'].to(device), large_data['depth'].to(device)], dim=1)
                    large_gt_normal = large_data['normal'].to(device)
                    large_gt_albedo = large_data['albedo'].to(device)

                    output = testnet(small_rgbd_input, large_rgbd_input)  # out_nor, out_alb, small_nor, small_alb
                    gt_list = [small_gt_normal,small_gt_albedo,large_gt_normal,large_gt_albedo]
                    loss, loss_nor, loss_alb, small_loss_nor, small_loss_alb = utils.small_large_loss(output,
                                                                                                      gt_list,
                                                                                                      alb_k, coarse_k)
                    ang_err_nor = utils.angular_loss(output[0], large_gt_normal)

                    test_loss_total += loss.item()
                    test_loss_nor += loss_nor.item()
                    test_loss_alb += loss_alb.item()
                    test_ang_err += ang_err_nor.item()

                print('epoch: %d,  Testing,   loss: %.3f  ' %
                      (epoch + 1, test_loss_total / len(testloader)))

                # log the running loss
                writer.add_scalar('Total test loss', test_loss_total/len(testloader), (epoch+1)*num_train_data)
                writer.add_scalar('Normal test loss', test_loss_nor/len(testloader), (epoch+1)*num_train_data)
                writer.add_scalar('Albedo test loss', test_loss_alb/len(testloader), (epoch+1)*num_train_data)
                writer.add_scalar('Normal test angular', test_ang_err / len(testloader), (epoch + 1) * num_train_data)
            testnet.train()

        if epoch % 1 == 0:  # save model every ? epoch
            torch.save(testnet.state_dict(), log_path+'/net_params.pth')


