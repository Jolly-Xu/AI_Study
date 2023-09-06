import torch.nn as nn
import torch
import numpy as np
import pickle as pk

# 参数
train_X = []
train_Y = []
padding_size = 10
UNK, PAD = '<', '>'


# 网络模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers, output_size, vocabSize, embeddingDim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.Lstm = nn.LSTM(input_size, hidden_size, layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    def forward(self, train_data):
        conf = torch.to(self.device)
        train_data = self.embedding(train_data)
        out, _ = self.lstm(train_data,conf)
        out = self.fc(out[-1, :, :])
        return out


# 数据预处理
# 读取本地训练数据
# 加载语料表构建数据集
with open("./data/vocab.pkl", "rb") as fp:
    pkl = pk.load(fp)

with open("./data/train.txt", encoding="utf-8") as f:
    k = 0
    for line in f:
        if k == 10:
            break
        train_arr = line.strip().split("\t")
        # 对文字进行分字处理
        temp = list(train_arr[0])
        if len(temp) >= padding_size:
            temp = temp[:padding_size]
        else:
            temp.extend((padding_size - len(temp)) * PAD)
        # 读取每一个字在字典中的value
        words_value = []
        for word in temp:
            w = pkl.get(word)
            w = 5000 if w is None else w
            words_value.append(w)
        try:
            train_X.append(torch.tensor(words_value, dtype=torch.float32))
        except TypeError:
            print(words_value)
        train_Y.append(torch.tensor(float(train_arr[1]), dtype=torch.float32))

# 数据转化为tensor
torch.stack(train_X)
torch.stack(train_Y)

# 读取词向量
embedding_predict = torch.tensor \
    (np.load("./data/embedding_Tencent.npz") \
         ["embeddings"].astype('float32'))

# 词向量大小
vocab_size = len(embedding_predict)
embedding_dim = len(embedding_predict[0])

inputSize = embedding_dim
hidden = 64
Lstm_layers = 1
outputSize = 10
batch_size = 64
epochs = 100
start = 0
total_all = len(train_X)

# 生成模型
model = LSTM(inputSize,hidden,Lstm_layers,outputSize,vocab_size,embedding_dim)

# 创建优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for i in epochs:
    # 还得补一个for循环
    end = start+batch_size*(i+1)
    end = end if end < total_all else total_all
    batch_data = train_X[start:end]
    start = end
    output = model(batch_data)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    




print(model)

