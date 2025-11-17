'''

@File:      bmp2jpg.py
@Author:    Lin Xuemin
@Date:      2024/12/23

'''

import cv2
import os

import numpy as np

# 指定包含BMP图像的文件夹路径
bmp_folder_path = r'Z:\05.广汽丰田\12.30采图\左后大灯后标识'

# 确保输出文件夹存在
jpg_folder_path = bmp_folder_path + '/jpg'
if not os.path.exists(jpg_folder_path):
    os.makedirs(jpg_folder_path)

# 遍历文件夹中的所有文件
for bmp_filename in os.listdir(bmp_folder_path):
    if bmp_filename.lower().endswith('.bmp'):
        # 构建完整的文件路径
        bmp_file_path = os.path.join(bmp_folder_path, bmp_filename)

        # 使用OpenCV读取BMP图像
        img = cv2.imdecode(np.fromfile(bmp_file_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img is not None:
            # 构建JPG文件的文件名
            jpg_filename = os.path.splitext(bmp_filename)[0] + '.jpg'
            jpg_file_path = os.path.join(jpg_folder_path, jpg_filename)

            # 将图像保存为JPG格式
            # cv2.imwrite(jpg_file_path, img)
            _, buf = cv2.imencode('.jpg', img)
            with open(jpg_file_path, 'wb') as f:
                f.write(buf)
            print(f'Converted {bmp_filename} to {jpg_filename}')
        else:
            print(f'Failed to read {bmp_filename}')