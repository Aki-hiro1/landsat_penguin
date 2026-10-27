from tools.calculate_tool import indices_generate
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio import features
import geopandas as gpd
import random


class ImgIdxDataset(Dataset):
    """
    遥感影像 + 点标签 数据集
    三分类：0=栖息地，1=背景，2=空类
    从 load_scenes 返回的 patches 列表加载数据
    实时计算13个指数通道
    支持数据增强（手动实现）
    """

    def __init__(self, patches, patch_size=128, augment=True, subset_indices=None):
        self.patch_size = patch_size
        self.augment = augment

        if subset_indices is not None:
            self.patches = [patches[i] for i in subset_indices]
        else:
            self.patches = patches

        print(f"加载完成: {len(self.patches)} 个图幅")

    def _apply_augmentations(self, img, indices, label, mask):
        """
        手动实现数据增强（同时对 img、indices、label、mask 做相同的几何变换）
        """
        H, W = img.shape[1], img.shape[2]

        # 1. 水平翻转
        if random.random() < 0.5:
            img = np.flip(img, axis=2).copy()
            indices = np.flip(indices, axis=2).copy()
            label = np.flip(label, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # 2. 垂直翻转
        if random.random() < 0.3:
            img = np.flip(img, axis=1).copy()
            indices = np.flip(indices, axis=1).copy()
            label = np.flip(label, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # 3. 随机旋转 90/180/270 度
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k, axes=(1, 2)).copy()
            indices = np.rot90(indices, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # # 4. 对6波段加高斯微扰
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        return img, indices, label, mask

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]

        # 1. 提取数据
        img = patch['image_data']  # [6, H, W]
        points = patch['points']
        H, W = img.shape[1], img.shape[2]

        # 2. 实时计算13个指数 [13, H, W]
        indices = indices_generate(img)

        # 3. 生成标签图（三分类：0=栖息地，1=背景，2=空类）
        label = np.full((H, W), 2, dtype=np.int64)  # 默认2（空类）
        rows = points['rows']
        cols = points['cols']
        lbls = points['labels']
        for r, c, l in zip(rows, cols, lbls):
            if 0 <= r < H and 0 <= c < W:
                label[r, c] = int(l)

        # 4. 生成损失掩码（只有有点的位置参与 loss）
        mask = np.zeros((H, W), dtype=np.float32)
        mask[rows, cols] = 1.0

        # 5. 裁剪到 patch_size
        if H != self.patch_size or W != self.patch_size:
            top = np.random.randint(0, H - self.patch_size + 1)
            left = np.random.randint(0, W - self.patch_size + 1)
            img = img[:, top:top+self.patch_size, left:left+self.patch_size]
            indices = indices[:, top:top+self.patch_size,
                              left:left+self.patch_size]
            label = label[top:top+self.patch_size, left:left+self.patch_size]
            mask = mask[top:top+self.patch_size, left:left+self.patch_size]

        # 6. 数据增强
        if self.augment:
            img, indices, label, mask = self._apply_augmentations(
                img, indices, label, mask
            )

        # 7. 转为 Tensor
        img = torch.from_numpy(img.astype(np.float32))
        indices = torch.from_numpy(indices.astype(np.float32))
        label = torch.from_numpy(label).long()
        mask = torch.from_numpy(mask)

        return {
            'image': img,
            'indices': indices,
            'label': label,
            'mask': mask,
            'scene_id': patch.get('scene_id', 'unknown')
        }


# ========== 测试 ==========
if __name__ == "__main__":
    from tools.file_tool import load_scenes
    patches = load_scenes("./config/context_manifest.yaml", 128, split='train')

    dataset = ImgIdxDataset(
        patches=patches,
        patch_size=128,
        augment=True
    )
    sample = dataset[0]
    print(f"Image: {sample['image'].shape}")
    print(f"Indices: {sample['indices'].shape}")
    print(f"Label: {sample['label'].shape}")
    print(f"Mask: {sample['mask'].shape}")

    label_np = sample['label'].numpy()
    mask_np = sample['mask'].numpy()
    unique, counts = np.unique(label_np[mask_np > 0.5], return_counts=True)
    print(f"点标签分布: {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"Mask中为1的位置数量: {(mask_np == 1).sum()}")
    print(f"标签唯一值: {np.unique(label_np)}")
