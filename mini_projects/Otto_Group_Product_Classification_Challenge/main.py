import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time


class GroupProduct(Dataset):
    def __init__(self, inputs, is_train):
        self.data = pd.read_csv(inputs)
        self.len = self.data.shape[0]
        if is_train:
            self.y_data = self.data["target"].str.replace("Class_", "").astype(int) - 1
            self.y_data = torch.tensor(self.y_data, dtype=torch.long)
            self.x_data = torch.tensor(
                self.data.drop(["target", "id"], axis=1).values, dtype=torch.float32
            )
        else:
            self.x_data = torch.tensor(
                self.data.drop("id", axis=1).values, dtype=torch.float32
            )
            self.id = self.data["id"]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(93, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 9),
        )

    def forward(self, x):
        return self.net(x)


model = Net()
dataset = GroupProduct("train.csv", True)
train_loader = DataLoader(dataset=dataset, batch_size=50, shuffle=True)
ctriterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
L = []
st = time.time()
for epoch in range(100):
    sum = 0
    for x, y in train_loader:
        y_pred = model(x)
        loss = ctriterion(y_pred, y)
        sum += loss.item() * x.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    L.append(sum / dataset.len)
print(time.time() - st)
plt.plot(L)
plt.xlabel("epoch")
plt.ylabel("CrossEntropyLoss")
plt.grid()
plt.show()
dataset = GroupProduct("test.csv", False)

# 切换为evaluate状态，这里不必要，
# 但是以后用到的一些函数可能会改变数据的分布结构，便于更好训练，
# 但是在测试阶段不能让它改变，
# 所以开启eval后它就不会进行train状态的转换了
model.eval()
# 注意：样例输出是0/1，不要以为就是输出判断结果，这种题一般输出概率，最后的分数是crossentropyloss
with torch.no_grad():
    y_pred = torch.softmax(model(dataset.x_data), dim=1)

submission = pd.DataFrame(
    y_pred.numpy(),
    columns=[
        "Class_1",
        "Class_2",
        "Class_3",
        "Class_4",
        "Class_5",
        "Class_6",
        "Class_7",
        "Class_8",
        "Class_9",
    ],
)
# plt.show()默认阻塞程序，也就是出现图时程序还没有运行到这里就提前停住了
# 所以你打开submission文件看到的只是旧文件
# 关上图之后就继续运行完程序，你就能看到输出文件变化了
submission.insert(0, "id", dataset.id)
submission.to_csv("submission.csv", index=False)
# 最终得分
# 0.76879
