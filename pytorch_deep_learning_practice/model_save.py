import torchvision
from torchvision.models import vgg16
import torch.nn as nn
import torch

# 方法一，保存模型结构和模型参数
model = vgg16()
torch.save(model, "vgg16_method1.pth")
# 方法二，保存模型参数
torch.save(
    model.state_dict(), "vgg16_method2.pth"
)  # state_dict()把模型参数转为字典形式
