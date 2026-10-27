import torch
import numpy as np
from pathlib import Path
import sys
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import geopandas as gpd
from rasterio.features import geometry_mask
import time

src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

try:
    from tools.file_tool import get_tif_path, load_image_by_projection
    from models.model import SE_UNet
    from tools.calculate_tool import indices_generate
except ImportError as e:
    print(f"导入失败: {e}")


# ========== 全局变量 ==========
SATELLITE_INDEX = 0

CHECKPOINT_PATH = r".\checkpoints\best_model_20261007_202940.pth"
INPUT_ROOT = r"d:\outputimg"
OUTPUT_DIR = r".\output\batch_results"

ROCK_SHP_PATH = r"C:\Users\39632\Desktop\dachuang\add_rockoutcrop_landsat_v7.3\add_rockoutcrop_landsat_v7.3.shp"
# ROCK_SHP_PATH = None

CENTER_X = 511720
CENTER_Y = 1715640
CROP_SIZE = 512
TILE_SIZE = 512


def convert_6ch_to_19ch(data_6ch):
    """6 波段 → 19 通道（6 原始 + 13 指数）"""
    _, H, W = data_6ch.shape
    data_6ch_flat = data_6ch.reshape(6, -1)
    indices = indices_generate(data_6ch_flat)
    data_19ch_flat = np.concatenate([data_6ch_flat, indices], axis=0)
    data_19ch = data_19ch_flat.reshape(19, H, W)
    return data_19ch.astype(np.float32)


def predict_image(model, image_data_19ch, device, tile_size=512):
    """分块预测"""
    model.eval()
    _, H, W = image_data_19ch.shape

    if H <= tile_size and W <= tile_size:
        img = image_data_19ch[:6, :, :]
        indices = image_data_19ch[6:, :, :]
        with torch.no_grad():
            img_tensor = torch.from_numpy(img[None]).to(device)
            indices_tensor = torch.from_numpy(indices[None]).to(device)
            output = model(img_tensor, indices_tensor)
            pred = output.argmax(dim=1).cpu().numpy()[0]
        return pred

    class_map = np.zeros((H, W), dtype=np.uint8)
    for row in range(0, H, tile_size):
        for col in range(0, W, tile_size):
            row_end = min(row + tile_size, H)
            col_end = min(col + tile_size, W)
            tile = image_data_19ch[:, row:row_end, col:col_end]
            img_tile = tile[:6, :, :]
            indices_tile = tile[6:, :, :]
            with torch.no_grad():
                img_tensor = torch.from_numpy(img_tile[None]).to(device)
                indices_tensor = torch.from_numpy(
                    indices_tile[None]).to(device)
                output = model(img_tensor, indices_tensor)
                pred = output.argmax(dim=1).cpu().numpy()[0]
            class_map[row:row_end, col:col_end] = pred
    return class_map


def get_shp_mask(shp_path, crs, shape, transform):
    """从 shp 生成掩膜"""
    gdf = gpd.read_file(shp_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    mask = geometry_mask(
        geometries=gdf.geometry,
        out_shape=shape,
        transform=transform,
        invert=True
    )
    return mask


def make_visualization(datas, class_map, rock_mask, output_png):
    """
    三栏可视化：原图 / 目标红色 / 分类+岩石掩膜
    类别 2（空类）当作背景（1）显示
    """
    nir = datas[3]
    green = datas[1]
    blue = datas[0]
    rgb_image = np.stack([nir, green, blue], axis=-1)

    # 把类别 2 当作背景（1）
    class_map_display = class_map.copy()
    class_map_display[class_map_display == 2] = 1

    # 右图：红色=栖息地(0)，蓝色=背景(1)
    overlay_right = rgb_image.copy()

    mask_target = (class_map_display == 0)
    overlay_right[mask_target, 0] = 1.0
    overlay_right[mask_target, 1] = 0.0
    overlay_right[mask_target, 2] = 0.0

    mask_background = (class_map_display == 1)
    overlay_right[mask_background, 0] = 0.0
    overlay_right[mask_background, 1] = 0.0
    overlay_right[mask_background, 2] = 1.0

    # 中图：目标红色，背景透明
    overlay_mid = rgb_image.copy()
    overlay_mid[mask_target, 0] = 1.0
    overlay_mid[mask_target, 1] = 0.0
    overlay_mid[mask_target, 2] = 0.0

    # 右图 + 岩石掩膜
    overlay_rock = overlay_right.copy()
    if rock_mask is not None:
        rock_alpha = 0.5
        overlay_rock[rock_mask, 0] = overlay_rock[rock_mask, 0] * \
            (1 - rock_alpha) + 0.0 * rock_alpha
        overlay_rock[rock_mask, 1] = overlay_rock[rock_mask, 1] * \
            (1 - rock_alpha) + 1.0 * rock_alpha
        overlay_rock[rock_mask, 2] = overlay_rock[rock_mask, 2] * \
            (1 - rock_alpha) + 1.0 * rock_alpha

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Image (NIR, Green, Blue)')
    axes[0].axis('off')

    axes[1].imshow(overlay_mid)
    axes[1].set_title('Target (Red), Background Transparent')
    axes[1].axis('off')
    legend_elements1 = [
        Patch(facecolor=[1, 0, 0], edgecolor='white',
              label='Target (Class 0)'),
        Patch(facecolor='none', edgecolor='white',
              label='Background (Class 1, Transparent)'),
    ]
    axes[1].legend(handles=legend_elements1,
                   bbox_to_anchor=(1.05, 1), loc='upper left')

    axes[2].imshow(overlay_rock)
    if rock_mask is not None:
        axes[2].set_title('Classification + Rock Mask (Cyan = Rock)')
        legend_elements2 = [
            Patch(facecolor=[1, 0, 0], edgecolor='white',
                  label='Target (Class 0)'),
            Patch(facecolor=[0, 0, 1], edgecolor='white',
                  label='Background (Class 1)'),
            Patch(facecolor=[0, 1, 1], edgecolor='white',
                  alpha=0.5, label='Rock Mask'),
        ]
    else:
        axes[2].set_title('Classification Result')
        legend_elements2 = [
            Patch(facecolor=[1, 0, 0], edgecolor='white',
                  label='Target (Class 0)'),
            Patch(facecolor=[0, 0, 1], edgecolor='white',
                  label='Background (Class 1)'),
        ]
    axes[2].axis('off')
    axes[2].legend(handles=legend_elements2,
                   bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PNG 已保存: {output_png}")


def process_one_folder(model, sub_folder, output_dir, device,
                       rock_shp_path=None):
    """处理单个子文件夹"""
    folder_name = sub_folder.name
    print(f"\n{'=' * 60}")
    print(f"处理文件夹: {folder_name}")

    tif_paths = get_tif_path(sub_folder, satellite_index=SATELLITE_INDEX)
    if tif_paths[0] is None:
        print(f"  跳过: 未找到有效 TIF")
        return

    # 固定中心点裁剪
    datas, profile, crs, shape = load_image_by_projection(
        tif_paths,
        center_x=CENTER_X,
        center_y=CENTER_Y,
        width=CROP_SIZE,
        height=CROP_SIZE
    )
    print(f"  影像尺寸: {datas.shape}")

    # 岩石掩膜
    rock_mask = None
    if rock_shp_path is not None and rock_shp_path.exists():
        rock_mask = get_shp_mask(
            rock_shp_path, crs, shape, profile['transform'])
        print(f"  岩石掩膜: {rock_mask.sum()} 个像素")

    datas_19ch = convert_6ch_to_19ch(datas)

    # 预测
    start_time = time.time()
    class_map = predict_image(model, datas_19ch, device, tile_size=TILE_SIZE)
    print(f"  预测耗时: {time.time() - start_time:.2f} 秒")

    unique, counts = np.unique(class_map, return_counts=True)
    print(f"  原始类别分布: {dict(zip(unique, counts))}")

    # 类别 2 当作背景（1）
    class_map[class_map == 2] = 1

    # 岩石掩膜过滤：岩石区域当成背景（类别 1）
    if rock_mask is not None:
        class_map[rock_mask] = 1

    unique, counts = np.unique(class_map, return_counts=True)
    print(f"  映射后类别分布: {dict(zip(unique, counts))}")

    # 输出 TIF
    output_tif = output_dir / f"{folder_name}_pred.tif"
    out_profile = profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255,
        'compress': 'deflate'
    })
    with rasterio.open(output_tif, 'w', **out_profile) as dst:
        dst.write(class_map, 1)
    print(f"  TIF 已保存: {output_tif}")

    # 输出 PNG
    output_png = output_dir / f"{folder_name}_vis.png"
    make_visualization(datas, class_map, rock_mask, output_png)


def main():
    print("=" * 60)
    print(f"卫星类型: {SATELLITE_INDEX}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"输入根目录: {INPUT_ROOT}")

    checkpoint_path = Path(CHECKPOINT_PATH)
    input_root = Path(INPUT_ROOT)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    rock_shp_path = Path(ROCK_SHP_PATH) if ROCK_SHP_PATH else None

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型（三分类）
    print("\n加载模型...")
    model = SE_UNet(
        in_channels_a=6,
        in_channels_b=13,
        num_classes=3,
        features=[64, 128, 256, 512],
        se_reduction=16
    ).to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载成功: {checkpoint_path}")

    # 遍历子文件夹
    sub_folders = [p for p in input_root.iterdir() if p.is_dir()]
    print(f"\n共找到 {len(sub_folders)} 个子文件夹")

    for sub_folder in sub_folders:
        try:
            process_one_folder(
                model=model,
                sub_folder=sub_folder,
                output_dir=output_dir,
                device=device,
                rock_shp_path=rock_shp_path
            )
        except Exception as e:
            print(f"  处理失败: {sub_folder.name}, 错误: {e}")
            continue

    print(f"\n全部完成，结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
