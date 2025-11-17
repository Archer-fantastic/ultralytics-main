import os
import xml.etree.ElementTree as ET

def convert_annotation(xml_path, txt_path, class_names):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(txt_path, 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in class_names:
                continue
            cls_id = class_names.index(cls)

            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)

            # 归一化
            x_center = (xmin + xmax) / 2.0 / w
            y_center = (ymin + ymax) / 2.0 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")



def xml2txt(xml_dir,txt_output_dir,class_names):
    # 创建输出文件夹
    os.makedirs(txt_output_dir, exist_ok=True)

    # 批量转换
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            txt_path = os.path.join(txt_output_dir, xml_file.replace('.xml', '.txt'))
            convert_annotation(xml_path, txt_path, class_names)

    print("转换完成！")


# 设置路径
xml_dir = r'D:\Projects\ultralytics-main\data\mask\annotations'       # XML 文件夹路径
txt_output_dir = r'D:\Projects\ultralytics-main\data\mask\yolo_txt'  # 输出 YOLO 标签路径
class_names = ['with_mask', 'without_mask']  # 替换为你的类别名

xml2txt(xml_dir,txt_output_dir,class_names)