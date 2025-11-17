# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

import cv2

from ultralytics import YOLO


def detect_pedestrians(video_path, output_path, conf=0.3):
    """
    用YOLOv8检测视频中的行人并保存结果
    :param video_path: 输入视频路径
    :param output_path: 输出视频路径
    :param conf: 置信度阈值.
    """
    # 加载YOLOv8模型（预训练模型，专门检测person等80类目标）
    model = YOLO("yolov8n.pt")  # "n"代表轻量化，速度快

    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return

    # 获取视频参数（用于保存输出）
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始处理视频：{os.path.basename(video_path)}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频处理完毕

        # 只检测行人（YOLOv8中person的类别ID为0）
        results = model.predict(
            frame,
            conf=conf,
            classes=[0],  # 仅检测person
            verbose=False,  # 关闭实时日志
        )

        # 绘制检测框（红色框标注行人）
        annotated_frame = results[0].plot(
            boxes=True,  # 显示检测框
            labels=True,  # 显示标签（person + 置信度）
        )

        # 保存帧到输出视频
        out.write(annotated_frame)

        # 实时显示（按ESC键退出）
        # cv2.imshow(f"YOLOv8 Pedestrian Detection: {os.path.basename(video_path)}", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    # 释放资源
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print(f"处理完成！结果保存至：{output_path}\n")


# ---------------------- 运行检测 ----------------------
if __name__ == "__main__":
    # 1. 处理普通视频（替换为你的普通视频路径）
    normal_video = r"D:\Min\Projects\VSCodeProjects\ultralytics-main\data\vedio\vedio (3).mp4"  # 下载的普通监控视频
    detect_pedestrians(
        video_path=normal_video,
        output_path=r"D:\Min\Projects\VSCodeProjects\ultralytics-main\results\vedio（3）_res2.mp4",
        conf=0.4,  # 置信度阈值，可根据效果调整（0.1-0.8）
    )

    # # 2. 处理360°鱼眼视频（替换为你的鱼眼视频路径）
    # fisheye_video = "fisheye_pedestrians.mp4"  # 下载的360°视频
    # detect_pedestrians(
    #     video_path=fisheye_video,
    #     output_path="fisheye_pedestrians_detected.mp4",
    #     conf=0.2  # 鱼眼视频边缘目标置信度可能较低，适当降低阈值
    # )
