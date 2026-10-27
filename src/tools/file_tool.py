import rasterio
import yaml
import numpy as np
from pathlib import Path
import geopandas as gpd
from rasterio.transform import rowcol


# ========== 全局变量 ==========
SATELLITE_INDEX = 0  # 0=Landsat, 1=Sentinel-2
MIN_POINTS_PER_PATCH = 20
CLASS_MAPPING = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
LABEL_FIELD = "id"


def get_tif_path(folder_path, satellite_index=SATELLITE_INDEX):
    """
    根据卫星类型找 6 个波段 TIF

    Args:
        folder_path: 影像文件夹
        satellite_index: 0=Landsat, 1=Sentinel-2

    Returns:
        image_paths: 6 个波段路径的列表
    """
    if satellite_index == 0:
        BAND_NAME = np.array(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    elif satellite_index == 1:
        BAND_NAME = np.array(["B2", "B3", "B4", "B8", "B11", "B12"])
    else:
        return [None] * 6
    BAND_INDEX = {name: i for i, name in enumerate(BAND_NAME)}

    folder_path = Path(folder_path)
    image_paths = [None] * 6
    tif_paths = [path for path in folder_path.iterdir() if path.is_file()
                 and path.suffix.lower() == '.tif']

    for path in tif_paths:
        band_name = path.stem.split('_')[-1]
        if satellite_index == 1:
            band_name = path.stem.split('_')[0]
        if band_name in BAND_NAME:
            image_paths[BAND_INDEX[band_name]] = path
    return image_paths


def load_image_by_projection(
    image_paths,
    center_x=None, center_y=None, width=None, height=None,
    norm_stats=None,
    return_stats=False,
    nodata_value=None,
):
    """
    根据投影坐标加载图像（适用于投影坐标系TIF）
    加载后 /10000，再按波段做 z-score 标准化（整张影像全局统计）

    :param image_paths: 图像路径列表
    :param center_x/center_y/width/height: 裁剪参数，None 表示整张
    :param norm_stats: 外部传入的固定统计量 {'mean': [C,1,1], 'std': [C,1,1]}
                       训练时传 None（自动计算并返回），推理时传入训练保存的值
    :param return_stats: 是否额外返回统计量
    :param nodata_value: NoData 值，若为 None 则尝试从 profile 读取
    :return: datas, profile, crs, shape [, stats]
    """
    datas = []
    nodata = nodata_value

    with rasterio.open(image_paths[0]) as src:
        if src.crs.is_geographic:
            raise ValueError(f"TIF不是投影坐标系，当前坐标系: {src.crs}")

        crs = src.crs
        profile = src.profile.copy()
        if nodata is None:
            nodata = src.nodata  # 可能为 None

        if center_x is None or center_y is None or width is None or height is None:
            window = None
            shape = (src.height, src.width)
        else:
            row, col = src.index(center_x, center_y)
            col_start = max(0, col - width // 2)
            row_start = max(0, row - height // 2)
            col_end = min(src.width, col_start + width)
            row_end = min(src.height, row_start + height)
            if col_end - col_start < width:
                col_start = max(0, col_end - width)
            if row_end - row_start < height:
                row_start = max(0, row_end - height)
            window = rasterio.windows.Window(
                col_start, row_start, col_end - col_start, row_end - row_start)
            transform = rasterio.windows.transform(window, src.transform)
            profile.update({'width': window.width, 'height': window.height,
                            'transform': transform})
            shape = (window.height, window.width)

    for path in image_paths:
        with rasterio.open(path) as src:
            datas.append(src.read(1, window=window))

    # datas = np.stack(datas, axis=0).astype(np.float32) / 10000  # [C, H, W]
    datas = np.stack(datas, axis=0).astype(np.float32) * 2.75e-5 + (-0.2)

    # # ============================================================
    # # 构建有效像素掩膜（排除 NoData / NaN / 非有限值）
    # # ============================================================
    # valid = np.isfinite(datas)
    # if nodata is not None:
    #     valid &= (datas != nodata / 10000)

    # # ============================================================
    # # z-score 标准化
    # # ============================================================
    # if norm_stats is not None:
    #     # 推理/验证：使用训练时固定的统计量
    #     mean = norm_stats['mean']
    #     std = norm_stats['std']
    # else:
    #     # 训练：按波段计算 mean/std，只用有效像素
    #     C = datas.shape[0]
    #     mean = np.zeros((C, 1, 1), dtype=np.float32)
    #     std = np.ones((C, 1, 1), dtype=np.float32)
    #     for c in range(C):
    #         vals = datas[c][valid[c]]
    #         if vals.size > 0:
    #             mean[c, 0, 0] = vals.mean()
    #             std[c, 0, 0] = vals.std()
    #     stats = {'mean': mean, 'std': std}

    # # 防止 std 为 0
    # std_safe = np.where(std > 1e-6, std, 1.0)
    # datas = (datas - mean) / std_safe

    # # 可选：把无效像素置 0（避免 NaN 进入模型）
    # datas = np.where(valid, datas, 0.0).astype(np.float32)

    if return_stats:
        return datas, profile, crs, shape
    return datas, profile, crs, shape


def map_labels(labels, mapping):
    """将原始类别映射为目标类别"""
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label not in mapping:
            raise KeyError(
                f"标签 {label} 不在映射字典中。可用: {list(mapping.keys())}")

    mapped = np.vectorize(mapping.get)(labels)
    return mapped.astype(np.int64)


def read_shp(shp_path):
    """读取点 shapefile"""
    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile 不存在: {shp_path}")

    gdf = gpd.read_file(shp_path)

    if not gdf.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
        raise ValueError(
            f"Shapefile 必须包含 Point 几何，当前类型: "
            f"{gdf.geometry.geom_type.unique()}")

    return gdf


def sample_by_class(rows, cols, raw_labels, sample_num):
    """
    按原始类别采样点，每类取 sample_num[i] 个

    Args:
        rows: 行号数组
        cols: 列号数组
        raw_labels: 原始类别数组
        sample_num: 每类采样数量列表，None 表示全部使用

    Returns:
        采样后的 rows, cols, raw_labels
    """
    if sample_num is None:
        return rows, cols, raw_labels

    selected_rows = []
    selected_cols = []
    selected_labels = []

    unique_classes = np.unique(raw_labels)

    for cls in unique_classes:
        mask = raw_labels == cls
        cls_rows = rows[mask]
        cls_cols = cols[mask]
        cls_labels = raw_labels[mask]

        n_available = len(cls_rows)
        n_target = sample_num[cls] if cls < len(sample_num) else n_available

        if n_target >= n_available:
            selected_rows.extend(cls_rows)
            selected_cols.extend(cls_cols)
            selected_labels.extend(cls_labels)
            print(f"    类别 {cls}: 可用 {n_available}，全部使用")
        else:
            indices = np.random.choice(
                n_available, size=n_target, replace=False)
            selected_rows.extend(cls_rows[indices])
            selected_cols.extend(cls_cols[indices])
            selected_labels.extend(cls_labels[indices])
            print(f"    类别 {cls}: 可用 {n_available}，采样 {n_target}")

    return (np.array(selected_rows), np.array(selected_cols),
            np.array(selected_labels))


def points_to_pixels(gdf, transform, image_height, image_width,
                     label_field=LABEL_FIELD, label_mapping=None,
                     return_fields=None, sample_num=None):
    """
    将点 shapefile 的坐标转换为像素行列号

    Args:
        gdf: GeoDataFrame
        transform: 影像的仿射变换
        image_height: 影像高度
        image_width: 影像宽度
        label_field: 类别字段名
        label_mapping: 类别映射字典
        return_fields: 要返回的字段列表
        sample_num: 每类采样数量列表，None 表示全部使用

    Returns:
        rows, cols, labels, field_values
    """
    x_coords = gdf.geometry.x.to_numpy()
    y_coords = gdf.geometry.y.to_numpy()

    rows, cols = rowcol(transform, x_coords, y_coords)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)

    in_bounds = (rows >= 0) & (rows < image_height) & \
                (cols >= 0) & (cols < image_width)

    rows_valid = rows[in_bounds]
    cols_valid = cols[in_bounds]

    if label_field not in gdf.columns:
        raise KeyError(
            f"找不到字段 '{label_field}'。可用字段: {gdf.columns.tolist()}")

    raw_labels = gdf[label_field].to_numpy()[in_bounds].astype(np.int64)

    dropped = int((~in_bounds).sum())
    if dropped > 0:
        print(f"⚠️ 丢弃了 {dropped} 个落在影像外的点")

    if len(rows_valid) == 0:
        raise ValueError("没有点落在影像范围内")

    print(f"  采样前: {len(rows_valid)} 个点")
    if sample_num is not None:
        rows_valid, cols_valid, raw_labels = sample_by_class(
            rows_valid, cols_valid, raw_labels, sample_num
        )
    print(f"  采样后: {len(rows_valid)} 个点")

    if label_mapping is not None:
        labels = map_labels(raw_labels, label_mapping)
    else:
        labels = raw_labels

    field_values = {}
    if return_fields:
        for field_name in return_fields:
            if field_name not in gdf.columns:
                raise KeyError(
                    f"找不到字段 '{field_name}'。"
                    f"可用字段: {gdf.columns.tolist()}")
            field_values[field_name] = gdf[field_name].to_numpy()[in_bounds]

    return rows_valid, cols_valid, labels, field_values


def load_scenes(manifest_path, patch_size=128,
                satellite_index=SATELLITE_INDEX, split=None):
    """
    从 manifest 文件读取场景，按 split 过滤，每个场景用自己的 sample_num

    Args:
        manifest_path: context_manifest.yaml 路径
        patch_size: 图幅尺寸
        satellite_index: 0=Landsat, 1=Sentinel-2
        split: 'train' / 'val' / 'test' / None
               None 表示返回所有场景

    Returns:
        patches: 图幅列表
    """
    manifest_path = Path(manifest_path)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    # 根目录
    root_dir = Path(manifest['root'])
    print(f"根目录: {root_dir}")

    scenes = manifest['scenes']

    # 按 split 过滤
    if split is not None:
        scenes = [s for s in scenes if s.get('split') == split]
        print(f"split='{split}', 场景数: {len(scenes)}")

    patches = []

    # 根据 satellite_index 选择波段名
    if satellite_index == 0:
        bands = ["B2", "B3", "B4", "B5", "B6", "B7"]
    elif satellite_index == 1:
        bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    else:
        bands = [None] * 6

    for entry in scenes:
        scene_id = entry['id']
        image_dir = Path(root_dir / entry['image_dir'])
        label_path = Path(root_dir / entry['label_path'])

        # 场景自己的 sample_num
        scene_sample_num = entry.get('sample_num', None)

        print(f"\n加载场景: {scene_id}")
        print(f"  split: {entry.get('split', 'unknown')}")
        print(f"  sample_num: {scene_sample_num}")

        # 1. 加载6通道整张影像
        tif_paths = get_tif_path(image_dir, satellite_index=satellite_index)
        tif_data, tif_profile, crs, shape = load_image_by_projection(tif_paths)
        H, W = shape
        print(f"  影像尺寸: {H}×{W}")

        # 2. 加载点SHP
        gdf = read_shp(label_path)
        if gdf.crs is not None and tif_profile.get('crs') is not None:
            if gdf.crs != tif_profile['crs']:
                print(f"  ⚠️ 重投影: {gdf.crs} → {tif_profile['crs']}")
                gdf = gdf.to_crs(tif_profile['crs'])

        # 3. 所有点转为像素坐标 + 类别映射（用场景自己的 sample_num）
        rows, cols, labels, infos = points_to_pixels(
            gdf,
            tif_profile['transform'],
            tif_profile['height'],
            tif_profile['width'],
            label_field=LABEL_FIELD,
            label_mapping=CLASS_MAPPING,
            return_fields=bands,
            sample_num=scene_sample_num
        )

        print(f"  总有效点: {len(rows)}")

        # 4. 统计类别分布
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  类别分布: {dict(zip(unique, counts))}")

        # 5. 对每个点，裁切图幅
        half = patch_size // 2
        patch_count = 0
        filtered_count = 0

        for i in range(len(rows)):
            center_r = int(rows[i])
            center_c = int(cols[i])
            center_label = int(labels[i])

            r_start = max(0, center_r - half)
            r_end = min(H, center_r + half)
            c_start = max(0, center_c - half)
            c_end = min(W, center_c + half)

            if r_end - r_start < patch_size or c_end - c_start < patch_size:
                continue

            patch_data = tif_data[:, r_start:r_end, c_start:c_end]

            patch_rows = []
            patch_cols = []
            patch_labels = []

            for j in range(len(rows)):
                r = int(rows[j])
                c = int(cols[j])
                if r_start <= r < r_end and c_start <= c < c_end:
                    patch_rows.append(r - r_start)
                    patch_cols.append(c - c_start)
                    patch_labels.append(labels[j])

            n_points = len(patch_rows)

            if center_label == 0 and n_points < MIN_POINTS_PER_PATCH:
                filtered_count += 1
                continue

            patches.append({
                'image_data': patch_data,
                'points': {
                    'rows': np.array(patch_rows),
                    'cols': np.array(patch_cols),
                    'labels': np.array(patch_labels)
                },
                'scene_id': scene_id,
                'center_row': center_r,
                'center_col': center_c,
                'center_label': center_label,
                'n_points': n_points
            })

            patch_count += 1

        print(f"  生成图幅: {patch_count} 个")
        print(f"  过滤掉(类别0点数<{MIN_POINTS_PER_PATCH}): "
              f"{filtered_count} 个")

    print(f"\n总图幅数: {len(patches)}")
    return patches
