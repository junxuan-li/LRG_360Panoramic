import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import glob
import numpy as np
from torchvision import transforms


class Real_Dataset(Dataset):
    """Real Dataset"""
    def __init__(self, root_dir, transformer=None):
        """
            root_dir (string): Directory with all the images.
              '/home/songzb/PycharmProjects/stereo_pano_relighting/data/real'
        """
        self.root_dir = root_dir
        self.room_list = sorted(glob.glob(os.path.join(root_dir, '*/')))
        self.transformer = transformer
        self.totensor = transforms.ToTensor()
        self.norm_rgb = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.room_list)

    def __getitem__(self, idx):
        room_path = self.room_list[idx]
        rgb_rawlight_path = os.path.join(room_path, 'up.png')
        depth_path = os.path.join(room_path, 'depth.npy')
        # normal_path = os.path.join(room_path, 'normal.png')
        # albedo_path = os.path.join(room_path, 'albedo.png')

        rgb_rawlight = Image.open(rgb_rawlight_path).convert('RGB')
        depth = np.load(depth_path)
        # normal = Image.open(normal_path).convert('RGB')
        # albedo = Image.open(albedo_path).convert('RGB')

        sample = {'rgb': self.norm_rgb(self.totensor(rgb_rawlight)), 'depth': self.totensor(depth).float()}
        if self.transformer is not None:
            sample = self.transformer(sample)
        return sample
