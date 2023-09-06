import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
import gzip
from torch import optim
import numpy as np

with gzip.open("data/mnist.pkl.gz", "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

print(x_train.shape)
print(y_train.shape)

print(x_valid.shape)
print(y_valid.shape)



x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape

loss_func = F.cross_entropy
bs = 64


# 构建模型
class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden01 = nn.Linear(784, 128)
        self.hidden02 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.hidden01(x))
        x = self.dropout(x)
        x = F.relu(self.hidden02(x))
        x = self.dropout(x)
        x = self.out(x)
        return x


train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)



def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2)
    )


def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for setp in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print("当前step:" + str(setp), '验证集损失：' + str(val_loss))



def get_model():
    model = Mnist()
    return model, optim.Adam(model.parameters(), lr=0.001)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(20,model,loss_func,opt,train_dl,valid_dl)
