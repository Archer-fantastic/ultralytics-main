# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

import cv2

from ultralytics import YOLO


def detect_keypoints(video_path, output_path, conf=0.3):
    """
    用YOLO关键点检测模型检测视频中行人的关键点并保存结果
    :param video_path: 输入视频路径
    :param output_path: 输出视频路径
    :param conf: 置信度阈值.
    """
    # 加载YOLOv8关键点检测模型（预训练的姿态估计模型）
    model = YOLO("yolov8n-pose.pt")  # "n"代表轻量化，"pose"表示关键点检测模型

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
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频处理完毕

        # 检测关键点（仅检测行人，类别ID为0）
        results = model.predict(
            frame,
            conf=conf,
            classes=[0],  # 仅检测行人
            verbose=False,  # 关闭实时日志
        )

        # 绘制检测结果：包含边界框、关键点和骨架连接
        annotated_frame = results[0].plot(
            boxes=True,  # 显示边界框
            labels=True,  # 显示标签（person + 置信度）
            kpt_radius=3,  # 关键点半径
            kpt_line=True,  # 显示关键点之间的连接（骨架）
        )

        # 保存帧到输出视频
        out.write(annotated_frame)

        # 显示进度
        frame_count += 1
        if frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"处理进度：{frame_count}/{total_frames} 帧 ({progress:.1f}%)")

        # 实时显示（按ESC键退出）
        # cv2.imshow("YOLOv8 Keypoint Detection", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    # 释放资源
    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print(f"处理完成！结果保存至：{output_path}\n")


# ---------------------- 运行关键点检测 ----------------------
if __name__ == "__main__":
    # 输入视频路径
    input_video = r"D:\Min\Projects\VSCodeProjects\ultralytics-main\data\vedio\vedio (2).mp4"
    # 输出视频路径
    output_video = r"D:\Min\Projects\VSCodeProjects\ultralytics-main\results\vedio（2）_keypoints.mp4"

    # 运行关键点检测
    detect_keypoints(
        video_path=input_video,
        output_path=output_video,
        conf=0.4,  # 置信度阈值，可根据效果调整
    )
