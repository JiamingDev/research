import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

batch_size = 64
# Compose,依次执行这几个操作
# Nomalize：把数据尽可能处理为均值为0，标准差为1，更利于神经网络训练的稳定性，
# 传元组是因为读入的图片可以有多通道，每个通道站一个位置
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
# 从官网下载数据到本地，train表明是下载训练集还是测试集，transform：对文件进行预处理
train_dataset = datasets.MNIST(
    root="./mnist/", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(
    root="./mnist/", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x):
        # 该数据图片像素尺寸为28*28=784，所以这样转后正好一行代表一张图片
        x = x.view(-1, 784)
        return self.net(x)


model = Net()
# 该函数可以完成
# 1.softmax(把最后得到的10个数转为概率，满足每个数>=0且和为1)，
# 2.取log
# 3.nllloss
criterion = torch.nn.CrossEntropyLoss()
# monetum:能够记忆历史梯度，并自动调节参数更新速度，
# 若某个参数连着朝一个方向更新了好几次，就加快它的更新速度
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train():
    for inputs, target in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            total += labels.size(0)
            # _存最大值，predicted存最大值的位置，均为一维，
            # torch.max对每行分别处理
            _, predicted = torch.max(outputs, dim=1)
            # 注意：tensor.sum()也是返回一个tensor，这是定义规定的
            # ==创建一个bool tensor，元素为每个比较结果，bool型，sum按照0/1统计
            correct += (predicted == labels).sum().item()
    print("accuracy on test set:%.2f %%" % (100 * correct / total))


for epoch in range(10):
    train()
    test()
