from ultralytics import YOLO
import argparse
import os

def parse_args():
    """解析命令行参数，支持灵活配置目标检测训练参数（含cache缓存配置）"""
    parser = argparse.ArgumentParser(description='YOLO 目标检测训练脚本（带缓存配置）')
    
    # 模型配置
    parser.add_argument('--model', type=str, 
                        default=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\weights\yolov3u.pt", 
                        help='预训练检测模型路径（如 yolov3u.pt, yolov8n.pt, yolo11s.pt）')
    
    # 数据集配置
    parser.add_argument('--data', type=str, 
                        default=r"D:\Min\Projects\VSCodeProjects\dataset\det_household-trash-recycling-dataset\data.yaml", 
                        help='数据集配置文件（data.yaml）路径')
    
    # 训练核心参数
    parser.add_argument('--epochs', type=int, default=10, 
                        help='训练轮次（小数据集30-100，大数据集100-300）')
    parser.add_argument('--imgsz', type=int, default=224, 
                        help='输入图片尺寸（需为32的倍数，检测常用640）')
    parser.add_argument('--batch', type=int, default=16, 
                        help='批次大小（GPU显存不足时减小，如8/4）')
    parser.add_argument('--device', type=str, default='0', 
                        help='训练设备（"0"为GPU，"cpu"为CPU，多GPU用"0,1"）')
    parser.add_argument('--lr0', type=float, default=0.01, 
                        help='初始学习率（默认0.01，小数据集可降至0.001）')
    
    # 缓存参数（核心新增）
    parser.add_argument('--cache', type=str, default='ram', 
                        help='数据集缓存方式：'
                             'True/\'ram\'=内存缓存（小数据集推荐），'
                             '\'disk\'=磁盘缓存（大数据集推荐），'
                             'False=不缓存（默认不缓存）')
    
    # 其他数据处理参数
    parser.add_argument('--workers', type=int, default=0, 
                        help='数据加载线程数（Windows建议0，Linux可设4-8）')
    
    # 输出配置
    parser.add_argument('--project', type=str, default='runs/detect', 
                        help='训练结果保存根目录')
    parser.add_argument('--name', type=str, default='train', 
                        help='本次训练文件夹名称（用于区分实验）')
    
    return parser.parse_args()

def train_detection(args):
    """训练YOLO目标检测模型（支持缓存配置）"""
    # 1. 加载预训练检测模型
    model = YOLO(args.model)
    if args.device != 'cpu':
        model.to(f'cuda:{args.device}' if args.device else 'cuda')
    print(f"已加载模型: {args.model}，训练设备: {args.device}")

    # 2. 检查数据集配置文件
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据集配置文件不存在: {args.data}")
    print(f"使用数据集配置: {args.data}")

    # 3. 解析缓存参数（转换为YOLO支持的格式）
    # 支持：True/'ram'（内存）、'disk'（磁盘）、False（不缓存）
    cache_mode = args.cache
    if cache_mode.lower() == 'true':
        cache_mode = True
    elif cache_mode.lower() == 'false':
        cache_mode = False
    print(f"数据集缓存方式: {cache_mode}（内存缓存适合小数据集，磁盘缓存适合大数据集）")

    # 4. 开始训练
    print("开始训练目标检测模型...")
    results = model.train(
        data=args.data,          # 数据集配置文件
        epochs=args.epochs,      # 训练轮次
        imgsz=args.imgsz,        # 输入图片尺寸
        batch=args.batch,        # 批次大小
        device=args.device,      # 训练设备
        lr0=args.lr0,            # 初始学习率
        workers=args.workers,    # 数据加载线程数
        cache=cache_mode,        # 缓存方式（核心参数）
        project=args.project,    # 结果保存根目录
        name=args.name,          # 训练文件夹名称
        
        # 训练模式配置
        pretrained=True,         # 使用预训练权重
        augment=True,            # 启用数据增强
        amp=False,               # 关闭AMP（稳定后可设为True）
        save=True,               # 保存训练结果
        val=True,                # 训练中自动验证
    )

    # 5. 验证集评估
    print("训练完成，开始在验证集上评估...")
    metrics = model.val()  # 输出mAP等指标
    print(f"验证集指标: {metrics}")

    # 6. 示例图片推理
    with open(args.data, 'r', encoding='utf-8') as f:
        import yaml
        data_cfg = yaml.safe_load(f)
        val_img_dir = os.path.join(os.path.dirname(args.data), data_cfg.get('val', ''))

    if os.path.exists(val_img_dir) and os.listdir(val_img_dir):
        sample_img = os.path.join(val_img_dir, os.listdir(val_img_dir)[0])
        print(f"对示例图片推理: {sample_img}")
        pred = model(sample_img)
        save_dir = os.path.join(args.project, args.name, 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        pred[0].save(os.path.join(save_dir, 'sample_pred.jpg'))
        print(f"推理结果已保存至: {save_dir}")

if __name__ == '__main__':
    args = parse_args()
    train_detection(args)