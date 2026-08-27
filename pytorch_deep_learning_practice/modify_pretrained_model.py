import torchvision
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn

vgg16_trained = vgg16(weights=VGG16_Weights.DEFAULT)
vgg16_NO = vgg16(weights=None)  # 默认就是None
# print(vgg16_trained.features[0].weight)  # type: ignore
# print(vgg16_trained)
# vgg16_trained.add_module("add_linear", nn.Linear(1000, 10))
# print(vgg16_trained)
# vgg16_trained.classifier.add_module("add_linear", nn.Linear(1000, 10))
# print(vgg16_trained)
vgg16_trained.classifier[6] = nn.Linear(4096, 10)
print(vgg16_trained)
