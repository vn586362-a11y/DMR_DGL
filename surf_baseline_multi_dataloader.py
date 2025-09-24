import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from lib.processing_utils import read_txt


class SURF_multi_train_loader(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.txt_path = os.path.join(root_path, 'train_list.txt')
        self.transform = transform
        self.lines = read_txt(self.txt_path)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index].strip().split(' ')
        rgb_img_path = os.path.join(self.root_path, line[0])
        depth_img_path = os.path.join(self.root_path, line[1])
        ir_img_path = os.path.join(self.root_path, line[2])
        label = int(line[3])

        img_rgb = Image.open(rgb_img_path).convert('RGB')
        img_depth = Image.open(depth_img_path).convert('RGB')
        img_ir = Image.open(ir_img_path).convert('RGB')

        if self.transform:
            img_rgb, img_depth, img_ir = self.transform(img_rgb), self.transform(img_depth), self.transform(img_ir)

        return (img_rgb, img_depth, img_ir), label


class SURF_multi_val_loader(Dataset):
    # === 核心修改：使其默认加载 val_list.txt ===
    def __init__(self, root_path, transform=None, list_file='val_list.txt'):
        self.root_path = root_path
        self.txt_path = os.path.join(root_path, list_file)
        self.transform = transform
        self.lines = read_txt(self.txt_path)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index].strip().split(' ')
        rgb_img_path = os.path.join(self.root_path, line[0])
        depth_img_path = os.path.join(self.root_path, line[1])
        ir_img_path = os.path.join(self.root_path, line[2])
        label = int(line[3])

        img_rgb = Image.open(rgb_img_path).convert('RGB')
        img_depth = Image.open(depth_img_path).convert('RGB')
        img_ir = Image.open(ir_img_path).convert('RGB')

        if self.transform:
            img_rgb, img_depth, img_ir = self.transform(img_rgb), self.transform(img_depth), self.transform(img_ir)

        return (img_rgb, img_depth, img_ir), label