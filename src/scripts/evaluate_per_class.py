import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.file_tool import load_scenes
    from tools.dataset import ImgIdxDataset
    from models.model import SE_UNet
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# ========== 与 train.py 保持一致 ==========
SATELLITE_INDEX = 0
PATCH_SIZE = 128
IN_CHANNELS_A = 6
IN_CHANNELS_B = 13
NUM_CLASSES = 3
FEATURES = [64, 128, 256, 512]
SE_REDUCTION = 16
IGNORE_INDEX = 2
BATCH_SIZE = 8
NUM_WORKERS = 4
MANIFEST_PATH = "./config/context_manifest.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {0: "类0", 1: "类1", 2: "空类(ignore)"}


def evaluate(model, loader, num_classes=3, ignore_index=2):
    """
    逐类统计 TP / FP / FN，最后算 precision / recall / IoU / F1。
    只统计 mask>0.5 且 label!=ignore_index 的像素。
    """
    # conf_mat[c][p] = 真实类 c 被预测为 p 的像素数
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="评估中"):
            img = batch['image'].to(DEVICE)
            indices = batch['indices'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)

            logits = model(img, indices)
            pred = logits.argmax(dim=1)  # (B, H, W)

            # 有效像素：mask>0.5 且 label != ignore_index
            valid = (mask > 0.5) & (label != ignore_index)

            lbl = label[valid].cpu().numpy()
            prd = pred[valid].cpu().numpy()

            # 累加到混淆矩阵（用 bincount 加速）
            idx = lbl * num_classes + prd
            counts = np.bincount(idx, minlength=num_classes * num_classes)
            conf_mat += counts.reshape(num_classes, num_classes)

    # 逐类指标
    results = {}
    total_valid = conf_mat.sum()
    for c in range(num_classes):
        if c == ignore_index:
            continue
        tp = conf_mat[c, c]
        fn = conf_mat[c, :].sum() - tp
        fp = conf_mat[:, c].sum() - tp
        support = conf_mat[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        ratio = support / total_valid if total_valid > 0 else 0.0

        results[c] = {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'f1': f1,
            'support': int(support),
            'ratio': ratio,
        }

    return results, conf_mat


def print_results(results, conf_mat, title=""):
    print("\n" + "=" * 78)
    if title:
        print(title)
    print("=" * 78)
    print(f"{'类别':<14}{'Precision':>10}{'Recall':>10}{'IoU':>10}"
          f"{'F1':>10}{'像素占比':>12}")
    print("-" * 78)
    for c, r in results.items():
        name = CLASS_NAMES.get(c, f"类{c}")
        print(f"{name:<14}{r['precision']:>10.4f}{r['recall']:>10.4f}"
              f"{r['iou']:>10.4f}{r['f1']:>10.4f}{r['ratio']:>11.2%}")
    print("-" * 78)
    # 宏平均
    macro_p = np.mean([r['precision'] for r in results.values()])
    macro_r = np.mean([r['recall'] for r in results.values()])
    macro_iou = np.mean([r['iou'] for r in results.values()])
    macro_f1 = np.mean([r['f1'] for r in results.values()])
    print(f"{'宏平均':<14}{macro_p:>10.4f}{macro_r:>10.4f}"
          f"{macro_iou:>10.4f}{macro_f1:>10.4f}")
    print("=" * 78)

    print("\n混淆矩阵 (行=真实类, 列=预测类):")
    print(conf_mat)


def main():
    ckpt_path = sys.argv[1] if len(
        sys.argv) > 1 else "./checkpoints/best_model_20261027_170637.pth"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"

    print(f"Checkpoint: {ckpt_path}")
    print(f"Split: {split}")
    print(f"Device: {DEVICE}")

    patches = load_scenes(
        MANIFEST_PATH, PATCH_SIZE,
        satellite_index=SATELLITE_INDEX, split=split
    )
    print(f"加载 {len(patches)} 个图幅")

    dataset = ImgIdxDataset(
        patches=patches, patch_size=PATCH_SIZE, augment=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    model = SE_UNet(
        in_channels_a=IN_CHANNELS_A,
        in_channels_b=IN_CHANNELS_B,
        num_classes=NUM_CLASSES,
        features=FEATURES,
        se_reduction=SE_REDUCTION
    ).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载模型 (epoch={checkpoint.get('epoch', '?')}, "
          f"best_loss={checkpoint.get('best_loss', '?'):.6f})")

    results, conf_mat = evaluate(model, loader,
                                 num_classes=NUM_CLASSES,
                                 ignore_index=IGNORE_INDEX)
    print_results(results, conf_mat, title=f"{split} 集逐类指标")


if __name__ == "__main__":
    main()
