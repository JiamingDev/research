import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TitanicDataset(Dataset):
    def __init__(self, inputs, is_train, scaler, columns):
        self.is_train = is_train
        self.train = pd.read_csv(inputs)
        self.passenger_id = self.train["PassengerId"]
        self.len = self.train.shape[0]
        # 处理PassengerId列：没有用处，直接扔
        self.train = self.train.drop("PassengerId", axis=1)  # axis=1删除这一列
        # 处理Name列：主要提取称谓，代表不同身份
        # 出现频率较低的，模型可能根本学不到规律或者发生过拟合
        self.rare_titles = [
            "Dr",
            "Rev",
            "Major",
            "Lady",
            "Countess",
            "Sir",
            "Jonkheer",
            "Col",
            "Capt",
        ]
        # 提取正则表达式，
        # 空格+一个或多个英文字母+.,
        # ():提取这一部分，
        # []:范围内字符提取，
        # +:符合前面字符的一个或多个，
        # \.:正则引擎解释为真正的.
        # r：转为raw string,防止python先把\解释为转义，报错
        self.train["Name"] = self.train["Name"].str.extract(r" ([A-Za-z]+)\.")
        # 替换rare_titles,防止过拟合
        self.train["Name"] = self.train["Name"].replace(self.rare_titles, "Rare")
        # One-Hot Encoding:有多少不同的东西就加多少新列，
        # 并把原列删除，新加的列为01编码，相当于把原来的不同东西看作不同特征
        # 然后这些新加列中每行只有一个1(这一行原先对应的那个东西那一列为1),
        # 其余均为0

        self.train = pd.get_dummies(self.train, columns=["Name"])
        # 处理Sex列：男性/女性
        self.train["Sex"] = self.train["Sex"].map({"male": 1, "female": 0})
        # 处理Age：有很多空的，也就是NaN
        self.train["Age"] = self.train["Age"].fillna(self.train["Age"].mean())
        # 处理Ticket列：按照每个票出现的次数分组，可能代表同行者
        # []代表分组完后只看这一列,transform('count')表示把这一列转为它出现的次数
        self.train["Ticket"] = self.train.groupby("Ticket")["Ticket"].transform("count")
        # 处理Cabin列：提取舱位,代表身份地位
        self.train["Cabin"] = self.train["Cabin"].fillna("U")  # 补缺失
        self.train["Cabin"] = self.train["Cabin"].str[0]
        self.train = pd.get_dummies(self.train, columns=["Cabin"])
        # 处理Embarked列：不同地方上船，也可能与社会阶层有关
        # mode()把空值填为众数
        self.train["Embarked"] = self.train["Embarked"].fillna(
            self.train["Embarked"].mode()[0]
        )
        self.train = pd.get_dummies(self.train, columns=["Embarked"])
        # 提取y
        # print(self.train.dtypes)#查看当前还有哪些列不是数字,发现有bool型
        # 把所有类型转化为float，保证后面正常转为torch，因为后面torch转化bool不能转为float32
        self.train = self.train.astype(float)
        if self.is_train:
            self.y_data = torch.tensor(
                self.train[["Survived"]].values, dtype=torch.float32
            )
            feature = self.train.drop("Survived", axis=1)
            self.column = feature.columns
        else:
            feature = self.train
            feature = feature.reindex(columns=columns, fill_value=0)
        # 把每一个连续特征分布转化为均值为0，标准差为1的数据，消除不同数据的范围差距，
        # 比如如果一组数据初始值明显比其它组大，
        # 那训练时可能会更偏向于训练这组数据，导致训练效果变差
        if scaler is None:
            self.scaler = StandardScaler()
            feature = self.scaler.fit_transform(feature)
        else:
            # 按照训练集的均值和标准差进行处理
            self.scaler = scaler
            feature = self.scaler.transform(feature)
        self.x_data = torch.tensor(feature, dtype=torch.float32)
        self.dim = self.x_data.shape[1]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Sequential:让输入以此经过这些函数，这里用ReLU而不用sigmoid是因为sigmoid很容易因为层数太深而梯度消失，训练缓慢
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 11),
            torch.nn.ReLU(),
            torch.nn.Linear(11, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 1),
        )

    def forward(self, x):
        return self.net(x)


dataset = TitanicDataset("train.csv", True, None, None)
model = Model(dataset.dim)
train_loader = DataLoader(dataset=dataset, batch_size=50, shuffle=True)
criterion = torch.nn.BCEWithLogitsLoss()  # 这个函数最后自动做sigmoid控制在[0,1]
# Adam自动调整内部学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
L = []
for epoch in range(1000):
    sum = 0
    for x, y in train_loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        sum += loss.item() * x.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    L.append(sum / dataset.len)
plt.plot(L)
plt.xlabel("epoch")
plt.ylabel("BCELoss")
plt.grid()
plt.show()
dataset = TitanicDataset("test.csv", False, dataset.scaler, dataset.column)
y_pred = model(dataset.x_data)
prob = torch.sigmoid(y_pred)
pred = (prob > 0.5).int()
pred = pred.numpy().reshape(-1)
submission = pd.DataFrame(
    {"PassengerId": dataset.passenger_id.values, "Survived": pred}
)
submission.to_csv("submission.csv", index=False)
