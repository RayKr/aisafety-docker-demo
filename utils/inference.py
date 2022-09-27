import os
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import json


def get_pic_from_dir(dir_path, transform):
    subPath = os.listdir(dir_path)
    if len(subPath) == 1 and subPath[0] != "images":
        dir_path = os.path.join(dir_path, subPath[0])
    if os.path.exists(os.path.join(dir_path, "inputs.npy")):
        samples = np.load(os.path.join(dir_path, "inputs.npy"))
        return samples
    elif os.path.exists(os.path.join(dir_path, "images/")):
        img_data = []
        dir_path += "/images"
        for file_name in os.listdir(dir_path):
            img = cv2.imread(os.path.join(dir_path, file_name))
            img = transform(img)
            img_data.append(img.numpy())
        return np.array(img_data)
    else:
        raise Exception(
            "The path {} do not has valid dataset {}".format(
                dir_path, os.listdir(dir_path)
            )
        )


# 读入系统提供的数据
# 注：default填写本地的路径并不影响平台，真正在平台运行评测的时候，会获取到真正的环境变量值
dataset = os.getenv("ENV_DATASET", default="datasets/demo/data")  # 基础数据集
c_dataset = os.getenv("ENV_CHILDDATASET", default="datasets/demo/cdata")  # 子数据集
save_path = os.getenv("ENV_RESULT", default="datasets/result")  # 中间结果存储路径
no = os.getenv("ENV_NO", default="0302")  # 结果文件的no
print(f"基础数据集：{dataset}")
print(f"子数据集：{c_dataset}")
print(f"结果文件路径：{save_path}")
print(f"结果编号：{no}")

result_dic = {"model": {}}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(model: nn.Module, checkpoint: str):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    model.to(device)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 推理原始图像
    origin_data = get_pic_from_dir(dataset, transform)
    with torch.no_grad():
        ret = model(torch.from_numpy(origin_data).float().to(device))
    result_dic["model"]["BDResult"] = ret.tolist()
    print("BDResult推理完成！")

    # 推理攻击样本
    for c_dir in os.listdir(c_dataset):
        if os.path.isdir(c_dataset + "/" + c_dir):
            child_data = get_pic_from_dir(os.path.join(c_dataset, c_dir), transform)
            with torch.no_grad():
                ret = model(torch.from_numpy(child_data).float().to(device))
            if "CDResult" not in result_dic["model"]:
                result_dic["model"]["CDResult"] = {}
            result_dic["model"]["CDResult"][c_dir] = ret.tolist()
    print("CDResult推理完成！")

    # 保存预测结果
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, no + ".json"), "w") as f:
        json.dump(result_dic, f)

    print("JSON Result saved.")
    return result_dic
