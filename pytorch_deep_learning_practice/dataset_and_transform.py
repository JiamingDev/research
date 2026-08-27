import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

trainset = torchvision.datasets.CIFAR10(
    "./dataset", True, transform=transforms.ToTensor(), download=True
)
testset = torchvision.datasets.CIFAR10(
    "./dataset", False, transform=transforms.ToTensor(), download=True
)
# print(testset[1])
# print(testset.classes)
# img, target = testset[1]
# print(img)
# print(target)
# print(testset.classes[target])
# writer = SummaryWriter("p10")
# for i in range(10):
#     img, _ = trainset[i]
#     writer.add_image("image2", img, i)
writer = SummaryWriter("log3")
testloader = DataLoader(testset, 64, True)

for epoch in range(2):
    step = 0
    for data in testloader:
        imgs, targets = data
        writer.add_images(f"images{epoch}", imgs, step)  # 这里用add_images（复数）
        step += 1
