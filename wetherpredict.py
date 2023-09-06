import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn import preprocessing

# 读取加载数据
features = pd.read_csv("data/temps.csv")
features = pd.get_dummies(features)
# print(features.head())
#
# print(type(features))

# 去除实际值这一列作为结果进行对比
labels = np.array(features['actual'])
features.drop('actual', axis=1)
features = np.array(features)
# 正则化 去均值 让数据均匀分布
input_features = preprocessing.StandardScaler().fit_transform(features)
# print(input_features[0])
# print(type(input_features))

# 把数据转化为tensor
# x = torch.tensor(input_features, dtype=float)
# y = torch.tensor(labels, dtype=float)

# 构建网络的参数
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 64

# 构建网络
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 64),
    torch.nn.Linear(64, output_size),
)
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)

losses = []
for i in range(2000):
    batch_loss = []
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if batch_size + start < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    if i % 50 == 0:
        print(i, np.mean(batch_loss))
