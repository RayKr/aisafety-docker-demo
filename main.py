from utils.inference import inference
import torchvision.models as models

# 要评测的模型和参数文件
model = models.resnet50(pretrained=False)
ckpt = "demo/models/resnet50_imagenet.pth"

# 执行推理，生成中间结果
inference(model, ckpt)
