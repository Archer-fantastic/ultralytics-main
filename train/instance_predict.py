# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import json
import os
from datetime import datetime
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO


class YOLOSegmentPredictor:
    def __init__(self, model_path, draw_chinese=False):
        """初始化实例分割预测器."""
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.draw_chinese = draw_chinese
        print(f"已加载实例分割模型：{model_path}，包含类别：{list(self.class_names.values())}")
        print(f"中文标签绘制：{'开启' if draw_chinese else '关闭'}")
        self.font = self._load_chinese_font() if draw_chinese else None

    def _load_chinese_font(self):
        """加载支持中文的字体."""
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/Library/Fonts/PingFang.ttc",
        ]
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, 30)
        print("警告：未找到中文字体，中文标签可能显示异常")
        return ImageFont.load_default()

    def _draw_annotations(self, img, boxes, masks, class_ids, confidences, img_shape):
        """绘制分割结果."""
        img_copy = img.copy()
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
        orig_h, orig_w = img_shape

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            mask = masks[i]
            class_id = class_ids[i]
            conf = confidences[i]
            class_name = self.class_names[class_id]
            color = colors[i % len(colors)]

            # 调整掩码尺寸
            mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask_3d = np.stack([mask_resized] * 3, axis=-1)

            # 绘制掩码
            img_copy = np.where(
                mask_3d, cv2.addWeighted(img_copy, 0.7, np.full_like(img_copy, color), 0.3, 0), img_copy
            )

            # 绘制边界框
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{class_name} ({conf:.2f})"
            label_y = max(30, y1)
            if self.draw_chinese:
                img_pil = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text((x1, label_y - 30), label, font=self.font, fill=(color[2], color[1], color[0]))
                img_copy = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(img_copy, label, (x1, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return img_copy

    def _mask_to_polygon(self, mask, img_shape):
        """
        将掩码转换为LabelMe格式的多边形坐标
        :param mask: 二值掩码（模型输出）
        :param img_shape: 原始图像尺寸 (height, width)
        :return: 多边形坐标列表 [[x1,y1], [x2,y2], ...].
        """
        orig_h, orig_w = img_shape
        # 调整掩码到原始图像尺寸
        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 查找轮廓
        contours, _ = cv2.findContours(
            mask_resized.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,  # 只保留最外层轮廓
            cv2.CHAIN_APPROX_SIMPLE,  # 简化轮廓
        )

        if not contours:
            return []

        # 取最大的轮廓（防止噪声）
        largest_contour = max(contours, key=cv2.contourArea)

        # 转换为坐标列表
        polygon = largest_contour.squeeze().tolist()
        # 确保是二维列表 [[x1,y1], [x2,y2], ...]
        if isinstance(polygon[0], int):
            polygon = [polygon]
        return polygon

    def _save_labelme_json(self, save_path, polygons, class_names, img_path, img_shape):
        """
        保存LabelMe格式的JSON文件
        :param save_path: JSON保存路径（不含后缀）
        :param polygons: 多边形坐标列表（每个实例一个多边形）
        :param class_names: 类别名称列表
        :param img_path: 原始图像路径
        :param img_shape: 图像尺寸 (height, width).
        """
        # 获取图像文件名
        img_filename = os.path.basename(img_path)

        # 构建LabelMe格式数据
        labelme_data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_filename,
            "imageData": None,  # 不保存图像二进制数据
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1],
            "image_path_list": [img_filename],
            "channels": 3,  # 默认RGB通道
        }

        # 添加每个实例的多边形信息
        for i, (polygon, class_name) in enumerate(zip(polygons, class_names)):
            if not polygon:
                continue  # 跳过空多边形

            shape = {
                "label": class_name,
                "points": polygon,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
            labelme_data["shapes"].append(shape)

        # 保存JSON
        with open(f"{save_path}.json", "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    def predict_single_image(self, img_path, save_dir=None):
        """单张图像实例分割预测."""
        if not os.path.exists(img_path):
            print(f"错误：图像不存在 - {img_path}")
            return None

        # 执行预测
        results = self.model(img_path)
        result = results[0]

        # 解析结果（CPU转换）
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        img_shape = result.orig_shape  # (height, width)

        # 终端输出
        instance_count = len(boxes)
        print(f"[{os.path.basename(img_path)}] 检测到 {instance_count} 个实例：")
        for i in range(instance_count):
            class_name = self.class_names[int(class_ids[i])]
            print(f"  实例 {i + 1}：类别={class_name}，置信度={confidences[i]:.4f}")

        # 保存结果
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            img_save_path = os.path.join(save_dir, f"{base_name}.jpg")
            json_save_path = os.path.join(save_dir, base_name)

            # 读取原图并绘制标注
            img = cv2.imread(img_path)
            if instance_count > 0 and self.draw_chinese:
                img = self._draw_annotations(img, boxes, masks, class_ids, confidences, img_shape)

            # 转换掩码为多边形（LabelMe格式）
            polygons = []
            class_names = []
            for i in range(instance_count):
                mask = masks[i] if i < len(masks) else None
                if mask is not None:
                    polygon = self._mask_to_polygon(mask, img_shape)
                    polygons.append(polygon)
                    class_names.append(self.class_names[int(class_ids[i])])

            # 保存图像和LabelMe格式JSON
            cv2.imwrite(img_save_path, img)
            self._save_labelme_json(json_save_path, polygons, class_names, img_path, img_shape)

        return {
            "image_path": img_path,
            "instance_count": instance_count,
            "instances": [
                {
                    "box": boxes[i].tolist(),
                    "class_id": int(class_ids[i]),
                    "class_name": self.class_names[int(class_ids[i])],
                    "confidence": float(confidences[i]),
                }
                for i in range(instance_count)
            ],
        }

    def predict_single_folder(self, folder_path, save_root=None, recursive=False):
        """单个文件夹实例分割预测."""
        if not os.path.isdir(folder_path):
            print(f"错误：文件夹不存在 - {folder_path}")
            return []

        img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        img_paths = []
        for ext in img_extensions:
            if recursive:
                img_paths.extend(glob(os.path.join(folder_path, "**", ext), recursive=True))
            else:
                img_paths.extend(glob(os.path.join(folder_path, ext)))

        if not img_paths:
            print(f"警告：文件夹中未找到图像 - {folder_path}")
            return []

        print(f"\n{'递归' if recursive else '非递归'}处理文件夹：{folder_path}，共{len(img_paths)}张图像")
        all_results = []
        for img_path in img_paths:
            save_dir = save_root
            if save_root:
                rel_path = os.path.relpath(os.path.dirname(img_path), folder_path)
                save_dir = os.path.join(save_root, rel_path)

            result = self.predict_single_image(img_path, save_dir)
            if result:
                all_results.append(result)

        return all_results


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="YOLO实例分割预测脚本（输出LabelMe格式JSON）")
    parser.add_argument(
        "--model",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\runs\segment\train2\weights\best.pt",
        help="训练好的分割模型路径",
    )
    parser.add_argument("--img", type=str, default=None, help="单张图像路径（可选）")
    parser.add_argument(
        "--folder",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\test_imgs\instance\img_dir",
        help="单个文件夹路径（可选）",
    )
    parser.add_argument("--recursive", action="store_false", help="是否递归读取文件夹")
    parser.add_argument(
        "--save",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\test_imgs",
        help="结果保存根目录",
    )
    parser.add_argument("--draw-chinese", action="store_true", help="是否绘制中文标签")
    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.img, args.folder]):
        print("错误：请指定预测对象（--img 或 --folder）")
        return

    save_root = None
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(args.save, f"segment_predict_{timestamp}")
        print(f"结果将保存至：{save_root}（图像+LabelMe格式JSON）")

    predictor = YOLOSegmentPredictor(args.model, draw_chinese=args.draw_chinese)

    if args.img:
        predictor.predict_single_image(args.img, save_root)

    if args.folder:
        predictor.predict_single_folder(args.folder, save_root, recursive=args.recursive)


if __name__ == "__main__":
    main()
