from utils.inference import inference
from models.model import get_model

# 要评测的模型和参数文件
model = get_model()
ckpt = "models/resnet50_imagenet.pth"

# 执行推理，生成中间结果
inference(model, ckpt)
