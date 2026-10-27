import numpy as np
from pathlib import Path
import sys
import rasterio
import matplotlib.pyplot as plt

src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

try:
    from tools.file_tool import get_tif_path, load_image_by_projection
except ImportError as e:
    print(f"导入失败: {e}")


# ========== 全局变量 ==========
SATELLITE_INDEX = 0  # landsat
TIF_FOLDER = r"D:\myProgramme\penguin_net-main\LandsatSample\img\wst20231207"


def main():
    print("=" * 60)
    print(f"TIF 文件夹: {TIF_FOLDER}")

    # 加载整张影像
    print("\n加载影像...")
    tif_folder = Path(TIF_FOLDER)
    tif_paths = get_tif_path(tif_folder, satellite_index=SATELLITE_INDEX)
    datas, profile, crs, shape = load_image_by_projection(tif_paths)
    print(f"影像尺寸: {datas.shape}")
    print(f"CRS: {crs}")
    print(f"Transform: {profile['transform']}")

    transform = profile['transform']

    # 构造 RGB 显示（注意：这里用的是 NIR/Green/Blue 假彩色）
    nir = datas[3]
    green = datas[1]
    blue = datas[0]
    rgb_image = np.stack([nir, green, blue], axis=-1)

    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb_image)
    ax.axis('off')

    def onclick(event):
        # 工具栏激活时，不触发点击
        if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode != '':
            return

        if event.xdata is None or event.ydata is None:
            return

        col_f = event.xdata
        row_f = event.ydata

        # 取整得到像素索引
        col = int(round(col_f))
        row = int(round(row_f))

        # 边界检查
        h, w = rgb_image.shape[:2]
        if not (0 <= row < h and 0 <= col < w):
            print(f"\n点击超出影像范围: (row={row}, col={col})")
            return

        # 投影坐标
        x_proj, y_proj = rasterio.transform.xy(
            transform, row, col, offset='center')

        # 该像素的 RGB（假彩色）值
        r_val, g_val, b_val = rgb_image[row, col]

        print(f"\n点击像素 (row={row}, col={col})")
        print(f"投影坐标: center_x={x_proj:.1f}, center_y={y_proj:.1f}")
        print(f"显示 RGB (NIR, Green, Blue): "
              f"({r_val:.4f}, {g_val:.4f}, {b_val:.4f})")
        print(f"对应波段原始值: "
              f"datas[3](NIR)={datas[3, row, col]:.4f}, "
              f"datas[1](Green)={datas[1, row, col]:.4f}, "
              f"datas[0](Blue)={datas[0, row, col]:.4f}")

        ax.plot(col, row, 'r+', markersize=12, markeredgewidth=2)
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()
    print("\n请点击图像查看投影坐标和 RGB 值，关闭窗口退出。")
    print("注意：使用工具栏（缩放/平移）时不会触发点击。")
    plt.show()


if __name__ == "__main__":
    main()
