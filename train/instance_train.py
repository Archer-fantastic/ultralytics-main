# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import os

from ultralytics import YOLO


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="YOLO 实例分割训练脚本")
    # 数据集配置
    parser.add_argument(
        "--data",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\dataset\instance_漏金属_lxm\yolo_instance_dataset\dataset.yaml",
        help="数据集配置文件（dataset.yaml）路径",
    )
    # 模型配置
    parser.add_argument(
        "--model",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\weights\yolov8n-seg.pt",
        help="预训练模型（如 yolov8n-seg.pt, yolov8s-seg.pt）",
    )
    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮次")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸（需为32的倍数）")
    parser.add_argument("--batch", type=int, default=16, help="批次大小（CPU训练可减小，如8）")
    parser.add_argument("--device", type=str, default="0", help="训练设备（cpu 或 0,1... 表示GPU）")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    # 输出配置
    parser.add_argument(
        "--project",
        type=str,
        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\train\runs\segment",
        help="训练结果保存根目录",
    )
    parser.add_argument("--name", type=str, default="train", help="本次训练文件夹名称")
    return parser.parse_args()


def train_segmentation(args):
    """训练YOLO实例分割模型."""
    # 1. 加载模型（预训练分割模型）
    model = YOLO(args.model)
    print(f"已加载模型: {args.model}")

    # 2. 检查数据集配置文件是否存在
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集配置文件不存在: {args.data}")
    print(f"使用数据集配置: {args.data}")

    # 3. 开始训练
    print("开始训练实例分割模型...")
    model.train(
        data=args.data,  # 数据集配置文件
        epochs=args.epochs,  # 训练轮次
        imgsz=args.imgsz,  # 输入图片尺寸
        batch=args.batch,  # 批次大小
        device=args.device,  # 训练设备
        lr0=args.lr0,  # 初始学习率
        project=args.project,  # 结果保存根目录
        name=args.name,  # 本次训练文件夹名称
        pretrained=True,  # 使用预训练权重（默认开启）
        augment=True,  # 启用数据增强
        mask_ratio=4,  # 掩码缩放比例（影响分割精度）
        save=True,  # 保存训练结果
        val=True,  # 训练中自动验证（默认开启）
        cache=True,
        amp=False,  # 关闭AMP，避免触发检查（后续可再开启）
    )

    # 4. 训练完成后在验证集上评估
    print("训练完成，开始在验证集上评估...")
    metrics = model.val()  # 评估结果包含AP、mask AP等指标
    print(f"验证集指标: {metrics}")

    # 5. 对示例图片进行推理（可选）
    # 找一张验证集图片作为示例
    val_img_dir = os.path.join(os.path.dirname(args.data), "images", "val")
    if os.path.exists(val_img_dir) and len(os.listdir(val_img_dir)) > 0:
        sample_img = os.path.join(val_img_dir, os.listdir(val_img_dir)[0])
        print(f"对示例图片推理: {sample_img}")
        pred = model(sample_img)
        # 保存推理结果（包含边界框和分割掩码）
        save_dir = os.path.join(args.project, args.name, "predictions")
        os.makedirs(save_dir, exist_ok=True)
        pred[0].save(os.path.join(save_dir, "sample_pred.jpg"))
        print(f"推理结果已保存至: {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    train_segmentation(args)
