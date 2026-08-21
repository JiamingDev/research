import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])
    def __getitem__(self,index):
        #返回元组
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x

if __name__=='__main__':#防止子进程继续创建子进程，造成runtimeerror
    model=Model()
    criterion=torch.nn.BCELoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
    L=[]
    dataset=DiabetesDataset('diabetes.csv')
    #num_worker：子进程数量，帮你加载数据，但不能进行训练
    train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
    for epoch in range(100):
        #inputs和labels为行数为batch_size的矩阵
        for i,(inputs,labels) in enumerate(train_loader):
            y_pred=model(inputs)
            loss=criterion(y_pred,labels)
            L.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    plt.plot(L)
    plt.xlabel('batch')
    plt.ylabel('BCELoss')
    plt.grid()
    plt.show()