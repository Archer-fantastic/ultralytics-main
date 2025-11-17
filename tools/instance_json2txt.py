# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# import os
# import json
# import shutil
# import random
# from tqdm import tqdm

# def labelme_to_yolo_seg(raw_dir, output_dir, train_ratio=0.8):
#     """
#     将LabelMe格式的实例分割数据转换为YOLO格式

#     参数:
#         raw_dir: 原始数据目录（包含.jpg和对应的.json文件）
#         output_dir: 输出YOLO格式数据的目录
#         train_ratio: 训练集占比
#     """
#     # 1. 初始化目录
#     img_train_dir = os.path.join(output_dir, 'images', 'train')
#     img_val_dir = os.path.join(output_dir, 'images', 'val')
#     lbl_train_dir = os.path.join(output_dir, 'labels', 'train')
#     lbl_val_dir = os.path.join(output_dir, 'labels', 'val')

#     os.makedirs(img_train_dir, exist_ok=True)
#     os.makedirs(img_val_dir, exist_ok=True)
#     os.makedirs(lbl_train_dir, exist_ok=True)
#     os.makedirs(lbl_val_dir, exist_ok=True)

#     # 2. 收集所有图片和对应的JSON文件
#     image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
#     all_images = [f for f in os.listdir(raw_dir) if f.lower().endswith(image_extensions)]
#     if not all_images:
#         print("未找到图片文件，请检查原始目录！")
#         return

#     # 3. 获取所有类别并分配ID（按字母顺序排序，确保ID固定）
#     classes = set()
#     for img_name in all_images:
#         json_name = os.path.splitext(img_name)[0] + '.json'
#         json_path = os.path.join(raw_dir, json_name)
#         if not os.path.exists(json_path):
#             print(f"警告：{img_name} 缺少对应标注文件 {json_name}，已跳过")
#             continue
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         for shape in data.get('shapes', []):
#             classes.add(shape['label'])
#     class_list = sorted(classes)  # 排序确保类别ID固定
#     class_id = {cls: i for i, cls in enumerate(class_list)}
#     print(f"检测到 {len(class_list)} 个类别：{class_list}")

#     # 4. 划分训练集和验证集
#     random.shuffle(all_images)
#     split_idx = int(len(all_images) * train_ratio)
#     train_images = all_images[:split_idx]
#     val_images = all_images[split_idx:]

#     # 5. 转换并复制文件
#     def process_images(images, img_dst, lbl_dst):
#         for img_name in tqdm(images, desc=f"处理{os.path.basename(img_dst)}"):
#             img_path = os.path.join(raw_dir, img_name)
#             json_name = os.path.splitext(img_name)[0] + '.json'
#             json_path = os.path.join(raw_dir, json_name)

#             # 跳过无标注的图片
#             if not os.path.exists(json_path):
#                 continue

#             # 复制图片到目标目录
#             shutil.copy2(img_path, os.path.join(img_dst, img_name))

#             # 解析JSON并生成YOLO标注
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#             img_h = data.get('imageHeight')
#             img_w = data.get('imageWidth')
#             if not img_h or not img_w:
#                 print(f"警告：{json_name} 缺少图片尺寸信息，已跳过")
#                 continue

#             # 生成.txt标注文件
#             lbl_name = os.path.splitext(img_name)[0] + '.txt'
#             lbl_path = os.path.join(lbl_dst, lbl_name)
#             with open(lbl_path, 'w', encoding='utf-8') as f:
#                 for shape in data.get('shapes', []):
#                     cls = shape['label']
#                     if cls not in class_id:
#                         continue  # 跳过未记录的类别
#                     points = shape['points']  # 原始坐标 (x, y)，未归一化
#                     # 归一化坐标（x / img_w, y / img_h）
#                     normalized = []
#                     for (x, y) in points:
#                         normalized.append(round(x / img_w, 6))  # 保留6位小数
#                         normalized.append(round(y / img_h, 6))
#                     # 写入格式：class_id x1 y1 x2 y2 ...
#                     f.write(f"{class_id[cls]} {' '.join(map(str, normalized))}\n")

#     # 处理训练集和验证集
#     process_images(train_images, img_train_dir, lbl_train_dir)
#     process_images(val_images, img_val_dir, lbl_val_dir)

#     # 6. 生成数据集配置文件（dataset.yaml）
#     yaml_path = os.path.join(output_dir, 'dataset.yaml')
#     with open(yaml_path, 'w', encoding='utf-8') as f:
#         f.write(f"path: {output_dir}\n")
#         f.write("train: images/train\n")
#         f.write("val: images/val\n")
#         f.write(f"nc: {len(class_list)}\n")
#         f.write(f"names: {class_list}\n")

#     print(f"转换完成！输出目录：{output_dir}")
#     print(f"训练集图片数：{len(train_images)}，验证集图片数：{len(val_images)}")
#     print(f"已生成配置文件：{yaml_path}")

# if __name__ == '__main__':
#     # -------------------------- 配置参数 --------------------------
#     raw_dir = r'D:\Min\Projects\VSCodeProjects\dataset\instance_251017_漏金属_lxm\data'  # 原始数据目录（存放.jpg和.json）
#     output_dir = r'D:\Min\Projects\VSCodeProjects\dataset\instance_251017_漏金属_lxm\yolo_instance_dataset'  # 输出YOLO格式目录
#     train_ratio = 0.8  # 训练集占比
#     # --------------------------------------------------------------

#     labelme_to_yolo_seg(
#         raw_dir=raw_dir,
#         output_dir=output_dir,
#         train_ratio=train_ratio
#     )

import json
import os
import random
import shutil

from tqdm import tqdm


def labelme_to_yolo_seg(raw_dir, output_dir, train_ratio=0.8):
    """将LabelMe格式的实例分割数据转换为YOLO格式（支持递归读取子文件夹）.

    参数:
        raw_dir: 原始数据根目录（会递归遍历所有子文件夹）
        output_dir: 输出YOLO格式数据的目录
        train_ratio: 训练集占比
    """
    # 1. 初始化输出目录
    img_train_dir = os.path.join(output_dir, "images", "train")
    img_val_dir = os.path.join(output_dir, "images", "val")
    lbl_train_dir = os.path.join(output_dir, "labels", "train")
    lbl_val_dir = os.path.join(output_dir, "labels", "val")

    os.makedirs(img_train_dir, exist_ok=True)
    os.makedirs(img_val_dir, exist_ok=True)
    os.makedirs(lbl_train_dir, exist_ok=True)
    os.makedirs(lbl_val_dir, exist_ok=True)

    # 2. 递归收集所有图片文件（支持子文件夹）
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG")
    all_images = []

    # 递归遍历所有子目录
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                # 保存图片的完整路径
                all_images.append(os.path.join(root, file))

    if not all_images:
        print("未找到任何图片文件，请检查原始目录！")
        return
    print(f"共发现 {len(all_images)} 张图片（含子文件夹）")

    # 3. 获取所有类别并分配ID（按字母顺序排序，确保ID固定）
    classes = set()
    valid_image_paths = []  # 存储有对应标注的图片路径

    for img_path in all_images:
        # 生成对应的JSON路径（与图片同目录、同名称）
        img_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        json_name = os.path.splitext(img_name)[0] + ".json"
        json_path = os.path.join(img_dir, json_name)

        if not os.path.exists(json_path):
            print(f"警告：{img_path} 缺少对应标注文件 {json_name}，已跳过")
            continue

        # 读取JSON并提取类别
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            for shape in data.get("shapes", []):
                classes.add(shape["label"])
            valid_image_paths.append(img_path)  # 只有有标注的图片才保留
        except Exception as e:
            print(f"警告：解析 {json_path} 失败，错误：{e!s}，已跳过")

    if not valid_image_paths:
        print("没有找到有效的带标注图片，转换终止！")
        return

    class_list = sorted(classes)  # 排序确保类别ID固定
    class_id = {cls: i for i, cls in enumerate(class_list)}
    print(f"检测到 {len(class_list)} 个类别：{class_list}")

    # 4. 划分训练集和验证集（基于有效图片）
    random.shuffle(valid_image_paths)
    split_idx = int(len(valid_image_paths) * train_ratio)
    train_images = valid_image_paths[:split_idx]
    val_images = valid_image_paths[split_idx:]

    # 5. 转换并复制文件
    def process_images(images, img_dst, lbl_dst):
        for img_path in tqdm(images, desc=f"处理{os.path.basename(img_dst)}集"):
            img_name = os.path.basename(img_path)
            img_dir = os.path.dirname(img_path)
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(img_dir, json_name)

            # 复制图片到目标目录（保留原始文件名，避免冲突）
            shutil.copy2(img_path, os.path.join(img_dst, img_name))

            # 解析JSON并生成YOLO标注
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            img_h = data.get("imageHeight")
            img_w = data.get("imageWidth")
            if not img_h or not img_w:
                print(f"警告：{json_path} 缺少图片尺寸信息，已跳过")
                continue

            # 生成.txt标注文件（与图片同名）
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path = os.path.join(lbl_dst, lbl_name)
            with open(lbl_path, "w", encoding="utf-8") as f:
                for shape in data.get("shapes", []):
                    cls = shape["label"]
                    if cls not in class_id:
                        continue  # 跳过未记录的类别
                    points = shape["points"]  # 原始坐标 (x, y)
                    # 归一化坐标（x / img_w, y / img_h）
                    normalized = []
                    for x, y in points:
                        normalized.append(round(x / img_w, 6))  # 保留6位小数
                        normalized.append(round(y / img_h, 6))
                    # 写入格式：class_id x1 y1 x2 y2 ...
                    f.write(f"{class_id[cls]} {' '.join(map(str, normalized))}\n")

    # 处理训练集和验证集
    process_images(train_images, img_train_dir, lbl_train_dir)
    process_images(val_images, img_val_dir, lbl_val_dir)

    # 6. 生成数据集配置文件（dataset.yaml）
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_list)}\n")
        f.write(f"names: {class_list}\n")

    print(f"\n转换完成！输出目录：{output_dir}")
    print(f"有效图片总数：{len(valid_image_paths)}")
    print(f"训练集图片数：{len(train_images)}，验证集图片数：{len(val_images)}")
    print(f"已生成配置文件：{yaml_path}")


if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    raw_dir = r"D:\Min\Projects\VSCodeProjects\dataset\instance_251017_漏金属_lxm\data"  # 原始数据根目录
    output_dir = r"D:\Min\Projects\VSCodeProjects\dataset\instance_251017_漏金属_lxm\yolo_instance_dataset"  # 输出目录
    train_ratio = 0.8  # 训练集占比
    # --------------------------------------------------------------

    labelme_to_yolo_seg(raw_dir=raw_dir, output_dir=output_dir, train_ratio=train_ratio)
