import os, random, json
import numpy as np
import torch, torchvision
from torchvision import transforms
from PIL import Image

# import torch.utils.data as Data

# 以下函数用于参数初始化


class SourceNotFoundExceptin(Exception):
    "this is user's Exception for check the length of name"

    def __init__(self, s):
        self.s = s

    def __str__(self):
        return "找不到{}!".format(self.s)


def processImage(Scale_ImageSize, Crop_ImageSize, imgDir, labelFile):
    """
    处理数据集
    :return:
    """

    labels_dict = {}
    with open(labelFile, "r", encoding="utf8") as fr:
        for line in fr:
            labels_dict[line.split(" ")[0]] = line.split(" ")[1]

    # 默认归一化的
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    mytransform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Scale(Scale_ImageSize),
            torchvision.transforms.CenterCrop(
                (
                    min(Crop_ImageSize[0], Scale_ImageSize[0]),
                    min(Crop_ImageSize[1], Scale_ImageSize[1]),
                )
            ),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )

    batch_size = len(labels_dict.values())
    labels = []
    xs = torch.empty(
        (
            batch_size,
            3,
            min(Crop_ImageSize[0], Scale_ImageSize[0]),
            min(Crop_ImageSize[1], Scale_ImageSize[1]),
        )
    )

    tmp = 0
    for name, label in labels_dict.items():
        img = Image.open(os.path.join(imgDir, name)).convert("RGB")
        # print(type(mytransform(img)),mytransform(img).shape)
        xs[tmp] = mytransform(img)
        labels.append(int(label.strip()))

    return xs, torch.tensor(labels)


def load_origin_dataset(dataPath, dataType="npy"):
    """
    加载原数据集
    :return:
    """
    # 用于解析下一层
    subPath = os.listdir(dataPath)
    if len(subPath) == 1:
        dataPath = os.path.join(dataPath, subPath[0])

    print("数据集类型：", dataType)

    if dataType == "npy":
        fName = os.path.join(dataPath, "inputs." + dataType)
        print(fName)
        if not os.path.exists(fName):
            raise SourceNotFoundExceptin("找不到数据集")

        samples = np.load(fName)
        xs = torch.from_numpy(samples).float()
        return xs

    if dataType == "json":
        fName = os.path.join(dataPath, "inputs." + dataType)
        print(fName)
        if not os.path.exists(fName):
            raise SourceNotFoundExceptin("找不到数据集")

        with open(fName, "w") as fw:
            samples_json = json.load(fw)
            samples = samples_json["data"]
            xs = torch.from_numpy(samples).float()
        return xs

    if dataType == "image":
        img_path = os.path.join(dataPath, "images")
        label_file = os.path.join(dataPath, "labels", "labels.txt")
        print(img_path, label_file)
        if not os.path.exists(img_path):
            raise SourceNotFoundExceptin("找不到数据集")

        xs, _ = processImage((224, 224), (224, 224), img_path, label_file)
        return xs

    return None


def load_adv_dataset(dataPath, dataType="npy"):
    """
    加载攻击数据集
    :param dataPath:
    :param dataType:
    :return:
    """

    # 用于解析下一层
    subPath = os.listdir(dataPath)
    if len(subPath) == 1:
        dataPath = os.path.join(dataPath, subPath[0])

    print("攻击数据集类型：", dataType)
    adv_dataset = {}
    for k in os.listdir(dataPath):
        print(k)
        advPath = os.path.join(dataPath, k)
        xs = load_origin_dataset(advPath, dataType)
        adv_dataset[k] = xs

    return adv_dataset


def saveAsJson(resultPath, result):
    with open(resultPath, "w") as fw:
        json.dump(result, fw)

    return


def npy2json(npyF, jsonF):
    """
    npy格式数据集转成json
    :return:
    """
    samples = np.load(npyF)
    # print(samples.reshape())
    # np.savetxt(txtF, samples, fmt='4d')
    with open(jsonF, "w") as fw:
        json.dump({"data": samples.tolist()}, fw)


if __name__ == "__main__":
    samples = npy2json(
        "C:/Users/mynam/Desktop/data/Datasets/test/FGSM_30_origin_inputs.npy",
        "C:/Users/mynam/Desktop/data/result/sample1.json",
    )
    npy2json(
        "C:/Users/mynam/Desktop/data/Datasets/test/FGSM_30_advs.npy",
        "C:/Users/mynam/Desktop/data/result/sample2.json",
    )
    print(samples)

    # with open('./test','w') as f:
    #     json.dump({"data":["1","2","3"]},f)
