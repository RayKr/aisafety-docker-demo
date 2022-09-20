from models.ResNet2 import getModel
from utils.inference import inference

# 要评测的模型和参数文件
model = getModel()
ckpt = "models/example.pt"

# 执行推理，生成中间结果
inference(model, ckpt)
