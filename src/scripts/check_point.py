import sys
from pathlib import Path
import yaml
import geopandas as gpd
import numpy as np

src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))


# ========== 全局变量 ==========
MANIFEST_PATH = "./config/context_manifest.yaml"
LABEL_FIELD = "id"


def main():
    manifest_path = Path(MANIFEST_PATH)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = yaml.safe_load(f)

    root_dir = Path(manifest['root'])
    print(f"根目录: {root_dir}")
    print("=" * 60)

    # 全局统计
    global_counts = {}

    for entry in manifest['scenes']:
        scene_id = entry['id']
        label_path = Path(root_dir / entry['label_path'])

        print(f"\n场景: {scene_id}")
        print(f"  shp: {label_path}")

        if not label_path.exists():
            print(f"  跳过: shp 不存在")
            continue

        gdf = gpd.read_file(label_path)
        print(f"  总点数: {len(gdf)}")

        if LABEL_FIELD not in gdf.columns:
            print(f"  跳过: 找不到字段 '{LABEL_FIELD}'")
            print(f"  可用字段: {gdf.columns.tolist()}")
            continue

        labels = gdf[LABEL_FIELD].to_numpy()
        unique, counts = np.unique(labels, return_counts=True)

        print(f"  类别分布:")
        for cls, count in zip(unique, counts):
            print(
                f"    类别 {cls}: {count} \t {count / gdf.shape[0] * 100:.2f}%")
            global_counts[cls] = global_counts.get(cls, 0) + count

    # 全局汇总
    print("\n" + "=" * 60)
    print("全局类别分布")
    print("=" * 60)

    if global_counts:
        total = sum(global_counts.values())
        for cls in sorted(global_counts.keys()):
            count = global_counts[cls]
            ratio = count / total * 100
            print(f"  类别 {cls}: {count} ({ratio:.2f}%)")
        print(f"  总计: {total}")
    else:
        print("  无数据")


if __name__ == "__main__":
    main()
