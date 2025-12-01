# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import os
from pathlib import Path

from tqdm import tqdm

names = []


def convert_poly_to_rect(coordinateList):
    X = [int(coordinateList[2 * i]) for i in range(int(len(coordinateList) / 2))]
    Y = [int(coordinateList[2 * i + 1]) for i in range(int(len(coordinateList) / 2))]

    Xmax = max(X)
    Xmin = min(X)
    Ymax = max(Y)
    Ymin = min(Y)
    flag = False
    if (Xmax - Xmin) == 0 or (Ymax - Ymin) == 0:
        flag = True
    return [Xmin, Ymin, Xmax - Xmin, Ymax - Ymin], flag


def convert_labelme_json_to_txt(json_path, img_path, out_txt_path):
    # json_list = glob.glob(json_path + '/*.json')
    # json_list = []
    # for json_file in os.listdir(json_path):
    #     if json_file.endswith(".json"):
    #         json_list.append(os.path.join(json_path, json_file))
    img_list = []
    for img_file in os.listdir(img_path):
        if img_file.endswith(".jpg"):
            img_list.append(os.path.join(img_path, img_file))

    for img_path in tqdm(img_list):
        last_dot_index = img_path.rfind(".")
        image_name = img_path[:last_dot_index]

        json_name = image_name + ".json"
        json_path = os.path.join(json_path, json_name)

        txt_name = image_name.split("\\")[-1] + ".txt"
        txt_path = os.path.join(out_txt_path, txt_name)
        if not os.path.exists(json_path):
            f = open(txt_path, "w", encoding="UTF-8")
            f.close()
            if "none" not in cls_cnt:
                cls_cnt["none"] = 1
            else:
                cls_cnt["none"] += 1
            continue

        with open(json_path, encoding="UTF-8") as f_json:
            json_data = json.loads(f_json.read())
        infos = json_data["shapes"]
        if len(infos) == 0:
            print(json_path, "is None!")
            continue
        img_w = json_data["imageWidth"]
        img_h = json_data["imageHeight"]

        # image_name = json_data['imagePath']
        # image_path = os.path.join(img_path, image_name)
        # if not os.path.exists(image_path):
        #     print(image_path, 'is None!')
        #     continue

        f = open(txt_path, "w", encoding="UTF-8")
        for label in infos:
            points = label["points"]
            if len(points) < 2:
                continue

            if len(points) == 2:
                x1 = points[0][0]
                y1 = points[0][1]
                x2 = points[1][0]
                y2 = points[1][1]
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                if len(points) < 4:
                    continue

            segmentation = []
            for p in points:
                segmentation.append(int(p[0]))
                segmentation.append(int(p[1]))

            bbox, flag = convert_poly_to_rect(list(segmentation))
            x1, y1, w, h = bbox

            if flag:
                continue

            x_center = x1 + w / 2
            y_center = y1 + h / 2
            norm_x = x_center / img_w
            norm_y = y_center / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            if label["label"] not in cls_dic:
                print(label["label"], txt_name)
                continue
            obj_cls = cls_dic[label["label"]]
            # if int(obj_cls) == 0: print(txt_name)
            cls_cnt[label["label"]] += 1

            line = [obj_cls, norm_x, norm_y, norm_w, norm_h]
            line = [str(ll) for ll in line]
            line = " ".join(line) + "\n"
            f.write(line)

        f.close()
    for cls in cls_cnt:
        print(f"{cls}的数量为：{cls_cnt[cls]}")


def process_one_dir(img_path):
    # img_path= r"E:\Dataset\3L_YZ_数据\01.一月份数据V1（0115-0120）\软管-颜色类\927W\[Q35]主燃油管\dataset"
    out_txt_path = rf"{img_path}_yolo_txt"
    json_path = img_path
    if len(list(Path(img_path).glob("*.jpg"))) > 0:
        if not os.path.exists(out_txt_path):
            os.makedirs(out_txt_path)
        convert_labelme_json_to_txt(json_path, img_path, out_txt_path)


def process_dirs(path):
    paths = list(Path(path).rglob("*/**"))
    # print(paths)
    for p in paths:
        if str(p).count("images_yolo_txt") + str(p).count("crop") > 0:
            continue
        print("processing:", p)
        process_one_dir(p)


"""
使用说明：修改以下内容
cls_dic：key是labelme里的标签名称，value就是yolo格式的txt文本里的类别。 例如：cls_dic = {"white": 0,"red":1}
json_path    
img_path     
out_txt_path 
"""
if __name__ == "__main__":
    cls_dic = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
    cls_cnt = {item: 0 for item in cls_dic}
    process_one_dir(r"U:\disk1\01.Datasets\05.lxm\01.广汽丰田\06.轮胎类\3L_YZ_轮胎_0121_0124_外圈")
