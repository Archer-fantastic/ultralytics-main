import shutil

from PIL import Image, ImageDraw
import json
import os
import random

from fontTools.ttLib.tables.C_P_A_L_ import Color


def random_color():
    # 生成一个随机颜色
    return tuple(random.randint(0, 255) for _ in range(3))


def json_to_image(json_path, image_size, label_to_color, mode='L'):
    # 读取JSON文件
    with open(json_path, 'r',encoding="utf-8") as f:
        data = json.load(f)

    # 创建一张空白图像，可以根据需要选择 'L' (灰度图) 或 'P' (8位彩色图)
    image = Image.new(mode, image_size, color='white')
    # if image.mode == "P":
    image = image.convert('P')
    draw = ImageDraw.Draw(image)

    # 将JSON中的标签信息映射到图像上
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        # 将多边形坐标映射到像素位置
        polygon_points = [(int(point[0]), int(point[1])) for point in points]

        # 获取标签对应的颜色，如果标签不存在于映射中，则生成一个新的颜色
        # color = label_to_color.setdefault(label, random_color())

        # 在图像上绘制多边形
        draw.polygon(polygon_points, fill= 128)

    return image


def batch_convert(json_folder, image_folder, output_folder, mode='L'):
    # 遍历JSON文件夹中的所有文件
    label_to_color = {0:(128,128,128),1:(128,0,128)}  # 存储标签到颜色的映射
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)

            # 构建对应的图像文件路径
            image_file = os.path.splitext(json_file)[0] + '.jpg'
            image_path = os.path.join(image_folder, image_file)

            # 转换JSON到图像
            result_image = json_to_image(json_path, Image.open(image_path).size, label_to_color, mode)

            # 保存结果图像
            output_path = os.path.join(output_folder, image_file.replace('.jpg', '.png'))
            result_image.save(output_path)
            shutil.copy(image_path, os.path.join(output_folder, image_file))


if __name__ == "__main__":
    # 设置文件夹路径
    json_folder = r'Z:\05.广汽丰田\20241203\林学民\一线数据\20241221\缝线类\948W\S17-2'
    image_folder = r'Z:\05.广汽丰田\20241203\林学民\一线数据\20241221\缝线类\948W\S17-2'
    output_folder = r'Z:\05.广汽丰田\20241203\林学民\一线数据\20241221\缝线类\948W\S17-2_dataset3'

    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # 批量转换为灰度图
    batch_convert(json_folder, image_folder, output_folder, mode='L')

    # 批量转换为8位彩色图
    # batch_convert(json_folder, image_folder, output_folder, mode='P')