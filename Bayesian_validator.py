import collections
import re


# 把语料库中的单词全部抽取出来，转成小写，并去除单词中间的特殊符号
def words(text): return re.findall('[a-z]+', text.lower())


# 统计语料库中个单词出现的次数
def train(features):
    model = collections.defaultdict(lambda: 1)  # 表示给所有key初始赋值其val为 1
    for f in features:
        model[f] += 1
    return model


NWORDS = train(words(open('./data/Bayesian.txt', encoding='utf-8').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'


# 编辑距离：
# 两个词之间的编辑距离定义为使用了几次插入(在词中插入一个单字母)，删除（删除一个字母）交换（交换两个相邻字母），替换（把一个字母替换成另一个字母），从而把一个词变成另一个词

# 返回所有与单词 w 编辑距离为 1 的集合
def edits1(word):
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion


# 返回所有与单词 w 编辑距离为 2 的集合  （就是把编辑距离=1的集合再计算一遍）
def known_edits2(word):
    return set(
        e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)  # set()函数可以创建一个无序不重复元素集  set还可以像数学上那样求交集和并集


def known(words):
    return set(w for w in words if w in NWORDS)


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    print(candidates)
    return max(candidates, key=lambda w: NWORDS[w])


# appl #appla #learw #tess #morw
correct('onln')
