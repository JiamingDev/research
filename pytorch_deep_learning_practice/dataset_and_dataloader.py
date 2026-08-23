import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        # 返回元组
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 注意这种多进程程序要在py文件里运行，不要在ipynb里面，因为ipynb里面谁是__main__无法确定，通常很乱
if __name__ == "__main__":  # 防止子进程继续创建子进程，造成runtimeerror
    model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    L = []
    dataset = DiabetesDataset("diabetes.csv")
    # num_worker：子进程数量，帮你加载数据，但不能进入这个更新逻辑
    # 注意这里后面要加persistent_workers=True，意思是第一轮创建的worker(进程)保留，
    # 下一轮循环时继续用，无需再创建新的worker，创建一个worker要几秒，如果每轮循环都
    # 重新创建，运行速度非常慢，要几十分钟，加入这个参数后速度明显加快，一分钟左右就行
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    for epoch in range(1000):
        # inputs和labels为行数为batch_size的矩阵
        sum = 0
        for i, (inputs, labels) in enumerate(train_loader):
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        L.append(sum / len(train_loader))
    plt.plot(L)
    plt.xlabel("epoch")
    plt.ylabel("BCELoss")
    plt.grid()
    plt.show()
