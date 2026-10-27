import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 路径设置
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.file_tool import load_scenes
    from tools.dataset import ImgIdxDataset
    from models.model import SE_UNet
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# ========== 全局变量 ==========
SATELLITE_INDEX = 0   # 0=Landsat, 1=Sentinel-2


# ========== 配置 ==========
class Config:
    MANIFEST_PATH = "./config/context_manifest.yaml"
    PATCH_SIZE = 128

    IN_CHANNELS_A = 6
    IN_CHANNELS_B = 13
    NUM_CLASSES = 3
    FEATURES = [64, 128, 256, 512]
    SE_REDUCTION = 16

    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4

    IGNORE_INDEX = 2

    SAVE_DIR = "./checkpoints"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss_with_mask(logits, labels, mask, ignore_index=2):
    """带掩码的交叉熵损失（三分类，空类不参与 loss）"""
    mask = mask.clone()
    mask[labels == ignore_index] = 0

    B, C, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = labels.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid_indices = mask_flat > 0.5
    if valid_indices.sum() == 0:
        # 保留计算图，避免 backward 报错
        return logits.sum() * 0.0

    logits_valid = logits_flat[valid_indices]
    labels_valid = labels_flat[valid_indices]

    loss = nn.functional.cross_entropy(logits_valid, labels_valid)
    return loss


def validate(model, val_loader, config):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            img = batch['image'].to(config.DEVICE)
            indices = batch['indices'].to(config.DEVICE)
            label = batch['label'].to(config.DEVICE)
            mask = batch['mask'].to(config.DEVICE)

            logits = model(img, indices)
            loss = compute_loss_with_mask(
                logits, label, mask, config.IGNORE_INDEX)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"检查点已保存: {filename}")


def create_data_loaders(train_patches, val_patches, test_patches, config):
    """从 yaml 的 split 直接构造 DataLoader"""
    train_scenes = set(p['scene_id'] for p in train_patches)
    val_scenes = set(p['scene_id'] for p in val_patches)
    test_scenes = set(p['scene_id'] for p in test_patches)

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_patches)} 图幅, 场景: {sorted(train_scenes)}")
    print(f"  验证集: {len(val_patches)} 图幅, 场景: {sorted(val_scenes)}")
    print(f"  测试集: {len(test_patches)} 图幅, 场景: {sorted(test_scenes)}")

    assert len(train_scenes & val_scenes) == 0, "训练集和验证集有场景重叠"
    assert len(train_scenes & test_scenes) == 0, "训练集和测试集有场景重叠"
    assert len(val_scenes & test_scenes) == 0, "验证集和测试集有场景重叠"
    print("  场景无重叠检查通过")

    train_dataset = ImgIdxDataset(
        patches=train_patches,
        patch_size=config.PATCH_SIZE,
        augment=True
    )
    val_dataset = ImgIdxDataset(
        patches=val_patches,
        patch_size=config.PATCH_SIZE,
        augment=False
    )
    test_dataset = ImgIdxDataset(
        patches=test_patches,
        patch_size=config.PATCH_SIZE,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        pin_memory=True)

    return train_loader, val_loader, test_loader


def train():
    """主训练函数"""
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("训练启动")
    print(f"卫星类型: {SATELLITE_INDEX}")
    print(f"设备: {config.DEVICE}")
    print(f"配置: {vars(config)}")

    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # 加载三个 split 的数据
    print("\n加载训练集...")
    train_patches = load_scenes(
        config.MANIFEST_PATH, config.PATCH_SIZE,
        satellite_index=SATELLITE_INDEX, split='train'
    )
    print("\n加载验证集...")
    val_patches = load_scenes(
        config.MANIFEST_PATH, config.PATCH_SIZE,
        satellite_index=SATELLITE_INDEX, split='val'
    )
    print("\n加载测试集...")
    test_patches = load_scenes(
        config.MANIFEST_PATH, config.PATCH_SIZE,
        satellite_index=SATELLITE_INDEX, split='test'
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        train_patches, val_patches, test_patches, config)

    print("\n创建模型...")
    model = SE_UNet(
        in_channels_a=config.IN_CHANNELS_A,
        in_channels_b=config.IN_CHANNELS_B,
        num_classes=config.NUM_CLASSES,
        features=config.FEATURES,
        se_reduction=config.SE_REDUCTION
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01)

    best_val_loss = float('inf')
    best_epoch = 0
    best_checkpoint_path = os.path.join(
        config.SAVE_DIR, f"best_model_{timestamp}.pth")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{config.EPOCHS}")
        for batch in progress_bar:
            img = batch['image'].to(config.DEVICE)
            indices = batch['indices'].to(config.DEVICE)
            label = batch['label'].to(config.DEVICE)
            mask = batch['mask'].to(config.DEVICE)

            logits = model(img, indices)
            loss = compute_loss_with_mask(
                logits, label, mask, config.IGNORE_INDEX)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate(model, val_loader, config)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{config.EPOCHS} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_val_loss,
                'config': vars(config)
            }, best_checkpoint_path)

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f} "
          f"(Epoch {best_epoch})")

    print("\n在测试集上评估...")
    checkpoint = torch.load(best_checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = validate(model, test_loader, config)
    print(f"测试集损失: {test_loss:.6f}")

    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train()
