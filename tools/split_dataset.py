"""
将数据集划分为训练集，验证集，测试集
"""
import os
from pathlib import Path
import random
import shutil
from os.path import isdir

from tqdm import tqdm


# 创建保存数据的文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def split_data(img_dir, label_dir, split_dir, TEST=True,split_ratio=[0.6,0.3,0.1],_Random=True):
    '''
    args:
        img_dir:原图片数据集路径
        label_dir:原yolog格式txt文件数据集路径
        split_dir:划分后数据集保存路径
        TEST:是否划分测试集
        用于将数据集划分为YOLO数据集格式的训练集,验证集,测试集
    '''
    random.seed(42)  # 随机种子
    # 1.确定原图片数据集路径
    datasetimg_dir = img_dir
    # 确定原label数据集路径
    datasetlabel_dir = label_dir

    images_dir = os.path.join(split_dir, 'images')
    labels_dir = os.path.join(split_dir, 'labels')
    dir_list = [images_dir, labels_dir]
    if TEST:
        type_label = ['train', 'val', 'test_yes']
    else:
        type_label = ['train', 'val']

    for i in range(len(dir_list)):
        for j in range(len(type_label)):
            makedir(os.path.join(dir_list[i], type_label[j]))

    # 3.确定将数据集划分为训练集，验证集，测试集的比例
    train_pct = split_ratio[0]
    valid_pct = 1-split_ratio[0]-split_ratio[2]
    test_pct = split_ratio[2]
    # 4.划分
    images = os.listdir(datasetimg_dir)
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = list(filter(lambda x: x.lower().endswith(valid_exts), images))
    # # 展示目标文件夹下所有的文件名
    # images = list(filter(lambda x: x.endswith('.jpg'), images))  # 取到所有以.txt结尾的yolo格式文件
    if _Random:
        random.shuffle(images)  # 乱序路径
    image_count = len(images)  # 计算图片数量
    train_point = int(image_count * train_pct)  # 0:train_pct
    valid_point = int(image_count * (train_pct + valid_pct))  # train_pct:valid_pct
    for i in tqdm(range(image_count), total=image_count, desc='Processing files'):
        if i < train_point:  # 保存0-train_point的图片和标签到训练集
            out_dir = os.path.join(images_dir, 'train')
            label_out_dir = os.path.join(labels_dir, 'train')
        elif train_point <= i < valid_point:  # 保存train_point-valid_point的图片和标签到验证集
            out_dir = os.path.join(images_dir, 'val')
            label_out_dir = os.path.join(labels_dir, 'val')
        else:  # 保存test_point-结束的图片和标签到测试集
            out_dir = os.path.join(images_dir, 'test')
            label_out_dir = os.path.join(labels_dir, 'test')
        if not os.path.exists(out_dir):os.makedirs(out_dir)
        if not os.path.exists(label_out_dir):os.makedirs(label_out_dir)

        last_dot_index = images[i].rfind('.')
        image_name = images[i][:last_dot_index]

        label_target_path = os.path.join(label_out_dir, image_name + '.txt')  # 指定目标保存路径
        label_src_path = os.path.join(datasetlabel_dir, image_name + '.txt')  # 指定目标原文件路径
        img_target_path = os.path.join(out_dir, images[i])  # 指定目标保存路径
        img_src_path = os.path.join(datasetimg_dir, images[i])  # 指定目标原图像路径

        if os.path.exists(img_target_path):
            img_target_path = os.path.join(out_dir, image_name + '_new' + images[i][last_dot_index+1:])
            label_target_path = os.path.join(label_out_dir, image_name + '_new.txt')  # 指定目标保存路径

        if os.path.exists(label_src_path) and os.path.exists(img_src_path):
            shutil.copy(label_src_path, label_target_path)  # 复制txt
            shutil.copy(img_src_path, img_target_path)  # 复制图片

    print('train:{}, val:{}, test_yes:{}'.format(train_point, valid_point - train_point,
                                               image_count - valid_point))

def split_dataset_by_dir(dir_path,dataset_path,TEST=False,split_ratio=[0.6,0.3,0.1],_Random=True):
    paths = list(Path(dir_path).rglob('*/**'))
    # print(paths)
    for p in paths:
        if str(p).count("images_yolo_txt") + str(p).count("crop")>0:continue
        print("processing:",p)

        label_dir = rf'{str(p)}_yolo_txt'
        split_data(p, label_dir, dataset_path, TEST=TEST,split_ratio=split_ratio,_Random=_Random)
        print(p.name + "已拷贝完成")
def split_one_dataset(img_dir,dataset_path,TEST=False,split_ratio=[0.6,0.3,0.1],_Random=True): 
        
        label_dir = rf'{img_dir}_yolo_txt'
        split_data(img_dir, label_dir, dataset_path, TEST=TEST,split_ratio=split_ratio,_Random=_Random)
if __name__ == "__main__":
    # split_dataset_by_dir(r"E:\Dataset\02.本田五厂\前悬工位\20250301_S33Z_分图\S33Z\其它类",
    #                      r"U:\disk1\01.Datasets\05.lxm\04.广汽本田五厂\其它类V1",
    #                      TEST=False,split_ratio=[0.6,0.4,0],_Random=True)
    split_one_dataset(r"D:\Projects\ultralytics-main\data\mask\images",
                      r"D:\Projects\ultralytics-main\data\mask_yolo",
                      split_ratio=[0.6,0.3,0.1])




