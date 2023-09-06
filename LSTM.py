import torch.nn as nn
import torch
import numpy as np
import pickle as pk

# 参数
train_X = [[]]
train_Y = []
padding_size = 10
UNK, PAD = '<UNK>', '<PAD>'


# 网络模型
class LSTM(nn.Module):
    def __int__(self, input_size, hidden_size, layers, output_size, vocabSize, embeddingDim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = layers
        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.Lstm = nn.LSTM(input_size, hidden_size, layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, train_data):
        h0 = torch.zeros(self.num_layers, train_data.size(1), self.hidden_size).to(train_data.device)
        c0 = torch.zeros(self.num_layers, train_data.size(1), self.hidden_size).to(train_data.device)
        out, _ = self.lstm(train_data, (h0, c0))
        out = self.fc(out[-1, :, :])
        return out


# 数据预处理
# 读取本地训练数据
# 加载语料表构建数据集
# with open("./data/vocab.pkl", "rb") as fp:
#     pkl = pk.load(fp)
#
# with open("./data/train.txt", encoding="utf-8") as f:
#     for line in f:
#         train_arr = line.strip().split("\t")
#         # 对文字进行分字处理
#         temp = list(train_arr[0])
#         if len(temp) >= padding_size:
#             temp = temp[:padding_size]
#         else:
#             temp.extend((padding_size - len(temp)) * PAD)
#         # 读取每一个字在字典中的value
#         words_value = []
#         for word in temp:
#             words_value.append(pkl.get(word))
#         train_X.append(words_value)
#         train_Y.append(train_arr[1])

# 读取词向量
embedding_predict = torch.tensor \
    (np.load("./data/embedding_Tencent.npz") \
         ["embeddings"].astype('float32'))

# 词向量大小
vocab_size = len(embedding_predict)
embedding_dim = len(embedding_predict[0])


