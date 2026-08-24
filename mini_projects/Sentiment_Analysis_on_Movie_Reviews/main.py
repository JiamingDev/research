import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import matplotlib.pyplot as plt

use_GPU = True
Batch_size = 256
Hidden_size = 100
N_layers = 2
N_class = 5


def createtensor(x):
    if use_GPU:
        device = torch.device("cuda:0")
        x = x.to(device)
    return x


class NameDataset(Dataset):
    def __init__(self, inputs, is_train, wordsdict):
        self.is_train = is_train
        self.data = pd.read_csv(inputs, sep="\t")
        self.len = self.data.shape[0]
        self.x_data = []
        if is_train:
            self.wordsdict = {}
            s = set()
            for _ in self.data["Phrase"]:
                _ = str(_)
                for word in _.split():
                    s.add(word)
            # pack_padded_sequence要求传入的seqlen中的值都严格大于0，也就是每行数据的长度大于0
            # 而且后面要embedding,要求数字都是非负整数，所以这里用0表示开头的空，1表示字典中没存的字符串
            # 做完该转换后一定要注意num_embedding要加2
            self.num = len(s) + 2
            for idx, word in enumerate(s, 2):
                self.wordsdict[word] = idx
        else:
            self.wordsdict = wordsdict
        for _ in self.data["Phrase"]:

            L = [0]
            _ = str(_)

            for word in _.split():
                L.append(self.wordsdict.get(word, 1))

            self.x_data.append(L)
        if is_train:
            self.y_data = torch.tensor(self.data["Sentiment"], dtype=torch.long)
        else:
            self.y_data = torch.zeros(self.len, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], len(self.x_data[index]), self.y_data[index]

    def __len__(self):
        return self.len


def my_collate(batch):
    inputs, seqlen, labels = zip(*batch)
    inputs = [torch.tensor(x, dtype=torch.long) for x in inputs]
    seqlen = torch.tensor(seqlen, dtype=torch.long)
    inputs = pad_sequence(inputs, True, 0)
    return inputs, seqlen, torch.tensor(labels, dtype=torch.long)


trainset = NameDataset("train.tsv", True, None)
trainloader = DataLoader(trainset, Batch_size, True, collate_fn=my_collate)
testset = NameDataset("test.tsv", False, trainset.wordsdict)
testloader = DataLoader(testset, Batch_size, False, collate_fn=my_collate)


def maketensor(inputs, seqlen, labels):
    seqlen, permidx = seqlen.sort(descending=True)
    inputs = inputs[permidx]
    labels = labels[permidx]
    return permidx, createtensor(inputs), seqlen, createtensor(labels)


class RNNclassifier(torch.nn.Module):
    def __init__(
        self, num_emb, emb_dim, hidden_size, num_layers, bidirectional, class_num
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirections = 2 if bidirectional else 1
        self.emb = torch.nn.Embedding(num_emb, emb_dim)
        self.gru = torch.nn.GRU(
            hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.linear = torch.nn.Linear(hidden_size * self.bidirections, class_num)

    def forward(self, permidx, x, seqlen, is_train):
        hidden = createtensor(
            torch.zeros(
                self.num_layers * self.bidirections, x.size(0), self.hidden_size
            )
        )
        x = self.emb(x)
        x = pack_padded_sequence(x, seqlen, batch_first=True)
        _, hidden = self.gru(x, hidden)
        if self.bidirections == 2:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden = hidden[-1]
        if not is_train:
            hidden = hidden[torch.argsort(permidx)]

        return self.linear(hidden)


classifier = RNNclassifier(
    trainset.num, Hidden_size, Hidden_size, N_layers, True, N_class
)
if use_GPU:
    classifier.to(torch.device("cuda:0"))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
L = []
for epoch in range(45):
    sum = 0
    for x, seqlen, y in trainloader:
        _, x, seqlen, y = maketensor(x, seqlen, y)
        out = classifier(_, x, seqlen, True)
        optimizer.zero_grad()
        loss = criterion(out, y)
        sum += loss.item() * len(x)
        loss.backward()
        optimizer.step()
    L.append(sum / len(trainset))
plt.plot(L)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()
L = []
for x, seqlen, y in testloader:
    with torch.no_grad():
        permidx, x, seqlen, y = maketensor(x, seqlen, y)
        out = classifier(permidx, x, seqlen, False)
        L.extend(out.max(dim=1)[1].tolist())
submission = pd.read_csv("sampleSubmission.csv", sep=",")
submission["Sentiment"] = L
submission.to_csv("submission.csv", index=False)
