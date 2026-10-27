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

CHECKPOINT_PATH = r".\checkpoints\best_model_20261027_203334.pth"
TIF_FOLDER = r"D:\myProgramme\penguin_net-main\LandsatSample\img\wst20201206"

# CENTER_X = 518709.5
# CENTER_Y = 2215412.1

CENTER_X = 422956.8
CENTER_Y = -1622584.5
CROP_SIZE = 512

ROCK_SHP_PATH = r"C:\Users\39632\Desktop\dachuang\add_rockoutcrop_landsat_v7.3\add_rockoutcrop_landsat_v7.3.shp"
# ROCK_SHP_PATH = None

OUTPUT_PNG = r".\output\visualization.png"
OUTPUT_TIF = r".\output\prediction.tif"


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
    print(f"可视化已保存: {output_png}")


def main():
    print("=" * 60)
    print(f"卫星类型: {SATELLITE_INDEX}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    output_png = Path(OUTPUT_PNG)
    output_tif = Path(OUTPUT_TIF)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载影像
    print("\n加载影像...")
    tif_folder = Path(TIF_FOLDER)
    tif_paths = get_tif_path(tif_folder, satellite_index=SATELLITE_INDEX)
    datas, profile, crs, shape = load_image_by_projection(
        tif_paths, CENTER_X, CENTER_Y, CROP_SIZE, CROP_SIZE
    )
    print(f"影像尺寸: {datas.shape}")

    # 岩石掩膜
    rock_mask = None
    if ROCK_SHP_PATH is not None and Path(ROCK_SHP_PATH).exists():
        print("生成岩石掩膜...")
        rock_mask = get_shp_mask(
            Path(ROCK_SHP_PATH), crs, shape, profile['transform'])
        print(f"岩石掩膜: {rock_mask.sum()} 个像素")

    # 6→19 通道
    print("转换 6→19 通道...")
    datas_19ch = convert_6ch_to_19ch(datas)
    print(f"19通道尺寸: {datas_19ch.shape}")

    # 加载模型（三分类）
    print("\n加载模型...")
    model = SE_UNet(
        in_channels_a=6,
        in_channels_b=13,
        num_classes=3,
        features=[64, 128, 256, 512],
        se_reduction=16
    ).to(device)

    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载成功: {checkpoint_path}")

    # 预测
    print("\n预测中...")
    start_time = time.time()
    class_map = predict_image(model, datas_19ch, device, tile_size=512)
    print(f"预测耗时: {time.time() - start_time:.2f} 秒")

    unique, counts = np.unique(class_map, return_counts=True)
    print(f"原始类别分布: {dict(zip(unique, counts))}")

    # 类别 2 当作背景（1）
    class_map[class_map == 2] = 1

    unique, counts = np.unique(class_map, return_counts=True)
    print(f"映射后类别分布: {dict(zip(unique, counts))}")

    # 保存 TIF（0=栖息地，1=背景）
    print("\n保存 TIF...")
    out_profile = profile.copy()
    out_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 255,
        'compress': 'deflate'
    })
    with rasterio.open(output_tif, 'w', **out_profile) as dst:
        dst.write(class_map, 1)
    print(f"TIF 已保存: {output_tif}")

    # 可视化 PNG
    print("\n生成可视化...")
    make_visualization(datas, class_map, rock_mask, output_png)


if __name__ == "__main__":
    main()
