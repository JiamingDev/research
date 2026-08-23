import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
from torch.nn.utils.rnn import pack_padded_sequence
import os  # operation system模块

os.chdir(r"D:\research\research\pytorch_deep_learning_practice")

print(
    os.getcwd()
)  # 获取当前工作目录，因为后面的打开文件是在当前工作目录中找，而不是py文件所在目录中找
use_GPU = True
N_char = 128
hidden_size = 100
num_layers = 2
# 问题：由名字拼写判断所属国家


# 本模型只取最后一层的最后一个hidden，因为这一个hidden已经包含了这个序列的所有信息，
# 而且这属于整个序列对应一个标签的问题，所以只取最后一个hidden
class NameDataset(Dataset):
    def __init__(self, is_train):
        filename = "train.csv" if is_train else "test.csv"
        with open(filename, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = {}
        for idx, country_name in enumerate(self.country_list, 0):
            self.country_dict[country_name] = idx
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len


trainset = NameDataset(True)
trainloader = DataLoader(trainset, batch_size=256, shuffle=True)
testset = NameDataset(False)
testloader = DataLoader(testset, batch_size=256, shuffle=False)


class RNNClassifier(torch.nn.Module):
    def __init__(
        self,
        num_embeddings,
        hidden_size,
        num_layers,
        output_size,
        bidirectional=True,
    ):
        super().__init__()
        self.bidirections = 2 if bidirectional else 1

        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.num_layers = num_layers
        self.emb = torch.nn.Embedding(
            embedding_dim=hidden_size, num_embeddings=num_embeddings
        )
        self.linear = torch.nn.Linear(hidden_size * self.bidirections, output_size)

    def forward(self, x, seqlen):

        x = x.t()
        hidden = createtensor(
            torch.zeros(
                self.num_layers * self.bidirections, x.size(1), self.hidden_size
            )
        )
        x = self.emb(x)
        # 这里操作完后待会进行gru时，gru会知道每个序列的真实长度，
        # 所以做完t->o->m后就得到最终hidden了，而不会因为后面有补全继续->pad->pad，导致逻辑错误
        # 同时减少不必要的计算
        # 因为这里的seqlen并不会参与矩阵运算，所以有些版本规定它应该在cpu上
        # 这个函数要求seqlen按照递减顺序，因为它要按照时间步逐渐调整
        x = pack_padded_sequence(x, seqlen)
        # print(x.data.shape)
        _, hidden = self.gru(x, hidden)
        if self.bidirections == 2:
            hidden = torch.cat(
                [hidden[-1], hidden[-2]], dim=1
            )  # 最后一层的两个方向的最终hidden拼接起来
        else:
            hidden = hidden[-1]  # 最后一层的最终hidden
        return self.linear(hidden)


def toListWithLen(name):
    return [ord(c) for c in name], len(name)


def createtensor(x):
    if use_GPU:
        device = torch.device("cuda:0")
        x = x.to(device)
    return x


def maketensor(names, countries):
    countries = countries.long()
    seq_and_lens = [toListWithLen(name) for name in names]
    nameList = [sl[0] for sl in seq_and_lens]
    seqlen = torch.tensor([sl[1] for sl in seq_and_lens], dtype=torch.long)
    seqlen, peridx = seqlen.sort(descending=True)
    peridx = peridx.tolist()
    nameList = [nameList[i] for i in peridx]
    countries = countries[peridx]
    seqtensor = torch.zeros(
        len(nameList),
        int(
            seqlen.max().item()
        ),  # 这里静态检查器会报错，它不知道item返回的具体是什么类型，所以显式转化
        dtype=torch.long,
    )
    for idx, (seq, leng) in enumerate(zip(nameList, seqlen)):
        seqtensor[idx, :leng] = torch.tensor(seq, dtype=torch.long)
    return (
        createtensor(seqtensor),
        seqlen,
        createtensor(countries),
    )


def train():
    # DataLoader中inputs为tuple,targets为tensor
    for inputs, targets in trainloader:
        # 后面的矩阵操作如embedding，gru必须要传规则矩阵
        input, seqlen, target = maketensor(inputs, targets)
        res = classifier(input, seqlen)
        loss = criterion(res, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test():
    correct = 0
    total = 0
    for inputs, targets in testloader:
        with torch.no_grad():
            input, seqlen, target = maketensor(inputs, targets)
            correct += classifier(input, seqlen).max(dim=1)[1].eq(target).sum().item()
            total += len(input)
    print(f"accuracy{correct/total}")


classifier = RNNClassifier(N_char, hidden_size, num_layers, trainset.country_num, True)

if use_GPU:
    device = torch.device("cuda:0")
    classifier.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
for epoch in range(100):
    train()
    test()
