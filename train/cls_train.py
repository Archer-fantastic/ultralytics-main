from ultralytics import YOLO
import argparse
import os

def parse_args():
    """解析命令行参数，支持灵活配置训练参数"""
    parser = argparse.ArgumentParser(description='YOLO 图像分类训练脚本')
    
    # 模型配置
    parser.add_argument('--model', type=str, 
                        default=r'D:\Min\Projects\VSCodeProjects\ultralytics-main\yolo11n-cls.pt', 
                        help='预训练分类模型路径（如 yolo11n-cls.pt, yolov8s-cls.pt）')
    
    # 数据集配置
    parser.add_argument('--data', type=str, 
                        default=r'D:\Min\Projects\VSCodeProjects\dataset\cls_OK-漏箔-气泡-脱碳\yolo_cls_dataset', 
                        help='数据集根目录（需包含train/val文件夹，每个类别一个子文件夹）')
    
    # 训练核心参数
    parser.add_argument('--epochs', type=int, default=50, 
                        help='训练轮次（小数据集30-100，大数据集可增加）')
    parser.add_argument('--imgsz', type=int, default=224, 
                        help='输入图片尺寸（需为32的倍数，如224/320/640）')
    parser.add_argument('--batch', type=int, default=16, 
                        help='批次大小（GPU显存不足时减小，如8/4）')
    parser.add_argument('--device', type=str, default='0', 
                        help='训练设备（"0"为GPU，"cpu"为CPU）')
    parser.add_argument('--lr0', type=float, default=0.01, 
                        help='初始学习率（默认0.01，小数据集可减小至0.001）')
    
    parser.add_argument('--cache', type=bool, default=True, 
                        help='是否加载入内存，提高训练速度，默认为True')
    
    
    # 输出配置
    parser.add_argument('--project', type=str, default='runs/classify', 
                        help='训练结果保存根目录')
    parser.add_argument('--name', type=str, default='train', 
                        help='本次训练文件夹名称（用于区分不同实验）')
    
    return parser.parse_args()

def train_classification(args):
    """训练YOLO图像分类模型"""
    # 1. 加载预训练分类模型
    # - 模型后缀为-cls，n/s/m/l/x代表尺寸（nano到xlarge）
    model = YOLO(args.model)
    print(f"已加载模型: {args.model}")

    # 2. 检查数据集目录是否存在
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集目录不存在: {args.data}")
    print(f"使用数据集: {args.data}")

    # 3. 开始训练
    print("开始训练图像分类模型...")
    results = model.train(
        data=args.data,          # 数据集根目录
        epochs=args.epochs,      # 训练轮次
        imgsz=args.imgsz,        # 输入图片尺寸
        batch=args.batch,        # 批次大小
        device=args.device,      # 训练设备
        lr0=args.lr0,            # 初始学习率
        project=args.project,    # 结果保存根目录
        name=args.name,          # 本次训练文件夹名称
        
        # 训练模式配置
        pretrained=True,         # 使用预训练权重（加速收敛，推荐开启）
        augment=True,            # 启用数据增强（提升泛化能力）
        amp=False,               # 关闭AMP（临时规避错误，稳定后可设为True）
        save=True,               # 保存训练结果
        val=True,                # 训练中自动验证

        cache=args.cache
    )

    # 4. 训练完成后在验证集上评估
    print("训练完成，开始在验证集上评估...")
    metrics = model.val()  # 评估指标包括准确率（top1、top5等）
    print(f"验证集指标: {metrics}")

    # 5. 对示例图片进行推理（可选）
    val_img_dir = os.path.join(args.data, 'val')  # 验证集图像根目录
    if os.path.exists(val_img_dir):
        # 找第一个类别文件夹中的第一张图片
        class_dirs = [d for d in os.listdir(val_img_dir) if os.path.isdir(os.path.join(val_img_dir, d))]
        if class_dirs:
            sample_img_path = os.path.join(val_img_dir, class_dirs[0])
            sample_imgs = [f for f in os.listdir(sample_img_path) if f.endswith(('.jpg', '.png'))]
            if sample_imgs:
                sample_img = os.path.join(sample_img_path, sample_imgs[0])
                print(f"对示例图片推理: {sample_img}")
                pred = model(sample_img)
                # 保存推理结果
                save_dir = os.path.join(args.project, args.name, 'predictions')
                os.makedirs(save_dir, exist_ok=True)
                pred[0].save(os.path.join(save_dir, 'sample_pred.jpg'))
                print(f"推理结果已保存至: {save_dir}")

if __name__ == '__main__':
    args = parse_args()
    train_classification(args)