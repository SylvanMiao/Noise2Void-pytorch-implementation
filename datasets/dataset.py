from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import copy

class Noise2VoidDataset(Dataset):
    def __init__(self, data_path, patch_size=(512, 512)):
        self.data_path = Path(data_path)
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
        self.paths = []
        for ext in exts:
            self.paths.extend(sorted(self.data_path.glob(ext)))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if img.mode in ('I;16', 'I;16B', 'I;16L'):
            arr = np.array(img, dtype=np.uint16).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            # normalize 16-bit to [0, 1]
            arr = arr / 65535.0
        else:
            arr = np.array(img.convert('L'), dtype=np.float32)
            arr = arr[:, :, None]
            arr = arr / 255.0

        source, bbox = self.random_crop(arr, self.patch_size)

        # multi-pixel blind-spot masking
        source, target, mask = self.blind_spot_mask(source)

        source = torch.from_numpy(source.transpose(2, 0, 1)).float()
        target = torch.from_numpy(target.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return source, target, mask, bbox, str(self.paths[idx])
      
      
    # 随机裁剪，这里准备用数据集的最大公约数512
    def random_crop(self, img, patch_size):        
        if type(patch_size) != tuple:
            raise TypeError('patch_size must be tuple')
        
        h, w, _ = img.shape

        if h == patch_size[0] and w == patch_size[1]:
            return img, (0, 0, h, w)
        if h < patch_size[0] or w < patch_size[1]:
            raise ValueError('patch_size must be <= image size')
        
        top = np.random.randint(0, h - patch_size[0])
        left = np.random.randint(0, w - patch_size[1])
        bottom = top + patch_size[0]
        right = left + patch_size[1]
        
        patch = img[top:bottom, left:right, :]
        return patch, (top, left, bottom, right)
    
    # 随机选择盲点并替换像素值
    def blind_spot_mask(self, img, blind_ratio=0.02, neighborhood_radius=5):
        # 获取图像的高度、宽度、通道数
        h, w, c = img.shape
        # 计算需要挖盲点的数量（至少1个），blind_ratio默认为0.02即2%
        n_blind = max(1, int(h * w * blind_ratio))

        # 生成所有像素位置索引 [0, 1, 2, ..., h*w-1]
        all_pos = np.arange(h * w)
        # 从所有像素位置中随机选择n_blind个不重复的位置
        chosen = np.random.choice(all_pos, n_blind, replace=False)
        # 将1D索引转换为2D坐标 (row, col)
        blind_h, blind_w = np.unravel_index(chosen, (h, w))

        # 创建图像的深拷贝，用于作为source输入
        source = copy.deepcopy(img)
        # 创建掩码矩阵，标记盲点位置，初始值为0
        mask = np.zeros((h, w, 1), dtype=np.float32)

        # 遍历每个盲点位置
        for bh, bw in zip(blind_h, blind_w):
            # 计算在盲点周围定义的邻域范围（行方向）：[r_min, r_max)
            r_min, r_max = max(0, bh - neighborhood_radius), min(h, bh + neighborhood_radius + 1)
            # 计算在盲点周围定义的邻域范围（列方向）：[c_min, c_max)
            c_min, c_max = max(0, bw - neighborhood_radius), min(w, bw + neighborhood_radius + 1)

            # 从邻域内随机选择一个行坐标
            nh = np.random.randint(r_min, r_max)
            # 从邻域内随机选择一个列坐标
            nw = np.random.randint(c_min, c_max)
            # 确保选中的邻域像素不是盲点本身
            while nh == bh and nw == bw:
                # 如果选中的是盲点本身，重新随机选择行坐标
                nh = np.random.randint(r_min, r_max)
                # 重新随机选择列坐标
                nw = np.random.randint(c_min, c_max)

            # 将source中盲点位置的像素替换为邻域内的随机像素值
            source[bh, bw, :] = img[nh, nw, :]
            # 在掩码中标记该盲点位置为1.0
            mask[bh, bw, :] = 1.0

        # 返回三个值：修改后的图像source、原始图像img作为target、记录盲点位置的mask
        return source, img, mask
      