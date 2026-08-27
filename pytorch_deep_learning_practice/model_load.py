import torch
import torchvision

# 对应方法一加载方式
model = torch.load(
    "vgg16_method1.pth", weights_only=False
)  # 默认为True，也就是只允许安全的加载权重类数据，但这个文件里面保存了完整的模型类对象，所以直接运行会报错
# print(model)
# 对应方法二
model = torchvision.models.vgg16()
model.load_state_dict(torch.load("vgg16_method2.pth"))
print(model)
