import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import glob
from torchvision import transforms


class PIL_Structured3D_Dataset(Dataset):
    """Structured3D Dataset"""

    def __init__(self, root_dir, transformer=None, training=True, split=0.9):
        """
            root_dir (string): Directory with all the images.
              '/data/Structured3D_dataset/Structured3D'
        """
        self.root_dir = root_dir
        all_room_list = sorted(glob.glob(os.path.join(root_dir, '*/2D_rendering/*/panorama')))
        if training:
            if split == 1:
                self.root_list = all_room_list
            else:
                self.room_list = all_room_list[:int(split * len(all_room_list))]
        else:
            self.room_list = all_room_list[int(split * len(all_room_list)):]
        self.transformer = transformer

    def __len__(self):
        return len(self.room_list)

    def __getitem__(self, idx):
        room_path = self.room_list[idx]
        rgb_rawlight_path = os.path.join(room_path, 'full/rgb_rawlight.png')
        depth_path = os.path.join(room_path, 'full/depth.png')
        normal_path = os.path.join(room_path, 'full/normal.png')
        albedo_path = os.path.join(room_path, 'full/albedo.png')

        rgb_rawlight = Image.open(rgb_rawlight_path).convert('RGB')
        depth = Image.open(depth_path)
        normal = Image.open(normal_path).convert('RGB')
        albedo = Image.open(albedo_path).convert('RGB')

        sample = {'rgb': rgb_rawlight, 'depth': depth, 'normal': normal, 'albedo': albedo}
        if self.transformer is not None:
            sample = self.transformer(sample)
        return sample

    def delete_data(self):
        return

    def reload_data(self):
        return


class ToTorchTensor(object):
    def __init__(self):
        self.func = transforms.ToTensor()

    def __call__(self, sample):
        for key in sample.keys():
            sample[key] = self.func(sample[key])
        return sample


class NormaliseTensors(object):
    def __init__(self):
        self.norm_rgb = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.norm_depth = transforms.Normalize([0, ], [1000, ])
        self.norm_normal = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __call__(self, sample):
        rgb_rawlight, depth, normal, albedo = sample['rgb'], sample['depth'], sample['normal'], sample['albedo']

        rgb_rawlight = self.norm_rgb(rgb_rawlight)
        albedo = self.norm_rgb(albedo)
        depth = self.norm_depth(depth.float())
        normal = self.norm_normal(normal)
        # normal = torch.nn.functional.normalize(normal,dim=0)   # This step is too slow in CPU

        sample['rgb'], sample['depth'], sample['normal'], sample['albedo'] = rgb_rawlight, depth, normal, albedo
        return sample


class AddGaussianNoise(object):
    def __init__(self, std=0.02, scale=1 / 4, input_size=(512, 1024)):
        self.std = std
        self.small_size = torch.Size([1, 1, int(input_size[0] * scale), int(input_size[1] * scale)])
        self.scale = scale

    def __call__(self, sample):
        depth = torch.unsqueeze(sample['depth'], dim=0)

        noise_map = torch.randn(self.small_size) * self.std + 1
        noise_bias = torch.randn(self.small_size) * self.std
        noise_ori = torch.randn(depth.size()) * self.std / 2
        depth_noise = depth * torch.nn.functional.interpolate(noise_map, scale_factor=1 / self.scale, mode='bicubic', align_corners=False) + \
                      torch.nn.functional.interpolate(noise_bias, scale_factor=1 / self.scale, mode='bicubic', align_corners=False) + \
                      noise_ori

        sample['depth'] = torch.squeeze(depth_noise, dim=0)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + ' Depth (std={0})'.format(self.std)


class AddGaussianNoise_RGB(object):
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, sample):
        noise = torch.randn(sample['rgb'].size()) * self.std
        sample['rgb'] += noise
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(std={0})'.format(self.std)


class AddSmallScaleData(object):
    def __init__(self, scale=4):
        self.scale = 1 / scale

    def __call__(self, sample):
        small_sample = dict()
        for key in sample.keys():
            temp = torch.unsqueeze(sample[key], dim=0)
            small_sample[key] = torch.squeeze(
                torch.nn.functional.interpolate(temp, scale_factor=self.scale, mode='nearest'), dim=0)
        out_sample = dict()
        out_sample['large'] = sample
        out_sample['small'] = small_sample
        return out_sample


class ScaleData(object):
    def __init__(self, scale=4):
        self.scale = 1 / scale

    def __call__(self, sample):
        for key in sample.keys():
            temp = torch.unsqueeze(sample[key], dim=0)
            sample[key] = torch.squeeze(
                torch.nn.functional.interpolate(temp, scale_factor=self.scale, mode='bilinear'), dim=0)

        return sample
