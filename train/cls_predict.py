# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import os
from datetime import datetime
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO


class YOLOClassifierPredictor:
    def __init__(self, model_path, draw_chinese=False):
        """
        初始化分类预测器
        :param model_path: 模型路径
        :param draw_chinese: 是否在图像上绘制中文标签（默认False）.
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # {id: 类别名}
        self.draw_chinese = draw_chinese  # 控制是否绘制中文标签
        print(f"已加载模型：{model_path}，包含类别：{list(self.class_names.values())}")
        print(f"中文标签绘制：{'开启' if draw_chinese else '关闭'}")

        # 预加载中文字体（仅当需要绘制时）
        self.font = self._load_chinese_font() if draw_chinese else None

    def _load_chinese_font(self):
        """加载支持中文的字体."""
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
            "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
            "/Library/Fonts/PingFang.ttc",  # macOS
        ]
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, 30)  # 字体大小30
        print("警告：未找到中文字体，中文标签可能显示异常")
        return ImageFont.load_default()

    def _draw_text(self, img, text, position=(10, 30)):
        """绘制文本（支持中文）."""
        if not self.draw_chinese:
            # 不绘制中文时使用OpenCV默认字体
            return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 绘制中文（使用PIL）
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=(0, 255, 0))  # 绿色文本
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def predict_single_image(self, img_path, save_root=None):
        """
        单张图像预测
        :param img_path: 图像路径
        :param save_root: 保存根目录（None则不保存）
        :return: 预测结果字典.
        """
        if not os.path.exists(img_path):
            print(f"错误：图像不存在 - {img_path}")
            return None

        # 执行预测
        results = self.model(img_path)
        result = results[0]
        top1_idx = result.probs.top1
        top1_conf = result.probs.top1conf.item()
        top1_class = self.class_names[top1_idx]

        # 终端输出结果
        print(f"[{os.path.basename(img_path)}] 类别：{top1_class}，置信度：{top1_conf:.4f}")

        # 保存结果（按类别分文件夹）
        if save_root:
            # 构建保存路径：save_root/类别/图像名
            class_save_dir = os.path.join(save_root, top1_class)
            os.makedirs(class_save_dir, exist_ok=True)

            # 读取图像并绘制标签
            img = cv2.imread(img_path)
            if self.draw_chinese:
                img = self._draw_text(img, f"{top1_class} ({top1_conf:.2f})")

            # 保存图像
            save_path = os.path.join(class_save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, img)
            # print(f"已保存至：{save_path}")  # 可选：打印保存路径

        return {"image_path": img_path, "class": top1_class, "confidence": top1_conf, "class_id": top1_idx}

    def predict_single_folder(self, folder_path, save_root=None, recursive=False):
        """
        单个文件夹预测
        :param folder_path: 文件夹路径
        :param save_root: 保存根目录（None则不保存）
        :param recursive: 是否递归读取子文件夹
        :return: 所有图像的预测结果列表.
        """
        if not os.path.isdir(folder_path):
            print(f"错误：文件夹不存在 - {folder_path}")
            return []

        # 递归/非递归获取所有图像路径
        img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        img_paths = []
        for ext in img_extensions:
            if recursive:
                # 递归查找所有子文件夹中的图像
                img_paths.extend(glob(os.path.join(folder_path, "**", ext), recursive=True))
            else:
                # 仅查找当前文件夹中的图像
                img_paths.extend(glob(os.path.join(folder_path, ext)))

        if not img_paths:
            print(f"警告：文件夹中未找到图像 - {folder_path}")
            return []

        print(f"\n{'递归' if recursive else '非递归'}处理文件夹：{folder_path}，共{len(img_paths)}张图像")
        all_results = []
        for img_path in img_paths:
            result = self.predict_single_image(img_path, save_root)
            if result:
                all_results.append(result)

        return all_results


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="YOLO分类预测脚本（按类别保存）")
    parser.add_argument(
        "--model",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\runs\classify\train\weights\best.pt",
        help="训练好的分类模型路径（best.pt）",
    )
    parser.add_argument("--img", type=str, default=None, help="单张图像路径（可选）")
    parser.add_argument(
        "--folder",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\test_imgs\cls\img_dir",
        help="单个文件夹路径（可选）",
    )
    parser.add_argument("--recursive", action="store_true", help="是否递归读取文件夹（配合--folder使用）")
    parser.add_argument(
        "--save",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\test_imgs",
        help="结果保存根目录（None则不保存）",
    )
    parser.add_argument("--draw-chinese", action="store_true", help="是否在图像上绘制中文标签（默认不绘制）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查输入参数（至少指定一种预测方式）
    if not any([args.img, args.folder]):
        print("错误：请指定预测对象（--img 单张图像 或 --folder 文件夹）")
        return

    # 处理保存目录（添加时间戳）
    save_root = None
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = os.path.join(args.save, f"predict_{timestamp}")
        print(f"结果将按类别保存至：{save_root}")

    # 初始化预测器（控制是否绘制中文）
    predictor = YOLOClassifierPredictor(args.model, draw_chinese=args.draw_chinese)

    # 单张图像预测
    if args.img:
        predictor.predict_single_image(args.img, save_root)

    # 单个文件夹预测（支持递归）
    if args.folder:
        predictor.predict_single_folder(args.folder, save_root, recursive=args.recursive)


if __name__ == "__main__":
    main()
