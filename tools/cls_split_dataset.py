# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import random
import shutil


def split_dataset(raw_data_root, output_root, train_ratio=0.8, img_extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """通用数据集划分：将所有图片按其直接父文件夹（类别名）划分到train/val.

    参数:
        raw_data_root: 原始数据根目录（所有图片的上层目录）
        output_root: 输出YOLO格式数据集目录
        train_ratio: 训练集占比（默认0.8）
        img_extensions: 图片文件扩展名
    """
    # 1. 收集所有图片路径及对应的类别（类别 = 图片所在的直接文件夹名称）
    all_images = []  # 存储元组 (图片路径, 类别名)

    # 递归遍历所有目录，寻找图片文件
    for root, dirs, files in os.walk(raw_data_root):
        for file in files:
            # 检查文件是否为图片
            if file.lower().endswith(img_extensions):
                img_path = os.path.join(root, file)
                cls_name = os.path.basename(root)  # 类别名 = 直接父文件夹名称
                all_images.append((img_path, cls_name))

    if not all_images:
        print("未找到任何图片，请检查原始数据路径！")
        return

    # 2. 按类别分组（确保每个类别的数据单独划分）
    from collections import defaultdict

    cls_to_images = defaultdict(list)
    for img_path, cls_name in all_images:
        cls_to_images[cls_name].append(img_path)

    # 3. 创建输出目录结构（train/类别名、val/类别名）
    train_root = os.path.join(output_root, "train")
    val_root = os.path.join(output_root, "val")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)

    # 为每个类别创建文件夹
    for cls_name in cls_to_images.keys():
        os.makedirs(os.path.join(train_root, cls_name), exist_ok=True)
        os.makedirs(os.path.join(val_root, cls_name), exist_ok=True)

    # 4. 划分并复制图片（按类别打乱后划分）
    train_count = 0
    val_count = 0
    for cls_name, img_paths in cls_to_images.items():
        # 打乱同一类别的图片顺序
        random.shuffle(img_paths)
        # 计算划分点
        split_idx = int(len(img_paths) * train_ratio)
        train_imgs = img_paths[:split_idx]
        val_imgs = img_paths[split_idx:]

        # 复制到训练集
        for img_path in train_imgs:
            dst_filename = os.path.basename(img_path)
            dst_path = os.path.join(train_root, cls_name, dst_filename)
            # 处理同名文件（避免覆盖，添加后缀）
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(dst_filename)
                dst_path = os.path.join(train_root, cls_name, f"{name}_dup{ext}")
            shutil.copy2(img_path, dst_path)  # 保留元数据
            train_count += 1

        # 复制到验证集
        for img_path in val_imgs:
            dst_filename = os.path.basename(img_path)
            dst_path = os.path.join(val_root, cls_name, dst_filename)
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(dst_filename)
                dst_path = os.path.join(val_root, cls_name, f"{name}_dup{ext}")
            shutil.copy2(img_path, dst_path)
            val_count += 1

    # 输出统计信息
    print("数据集划分完成！")
    print(f"总类别数: {len(cls_to_images)}")
    print(f"总图片数: {len(all_images)}")
    print(f"训练集: {train_count} 张（{train_ratio * 100}%）")
    print(f"验证集: {val_count} 张（{(1 - train_ratio) * 100}%）")
    print(f"输出路径: {output_root}")


if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    raw_data_root = r"D:\Min\Projects\VSCodeProjects\dataset\cls_dataset3\data"  # 替换为你的原始数据根目录
    output_root = (
        r"D:\Min\Projects\VSCodeProjects\dataset\cls_dataset3\data\yolo_cls_dataset"  # 输出的YOLO格式数据集目录
    )
    train_ratio = 0.8  # 训练集占比（可调整为0.7、0.9等）
    # --------------------------------------------------------------

    split_dataset(raw_data_root=raw_data_root, output_root=output_root, train_ratio=train_ratio)
