"""语言模型数据集 - 序列数据采样方法

本模块实现了两种序列数据采样方法:
1. 随机采样: 打破相邻小批量之间的时序关联
2. 顺序分区: 保持相邻小批量之间的时序关联

这些采样方法用于训练循环神经网络(RNN)等序列模型.
"""

import torch
from d2l import torch as d2l
from d2l_utils import (
    read_time_machine, 
    seq_data_iter_random, 
    seq_data_iter_sequential,
    SeqDataLoader,
    load_data_time_machine
)

# ==================== 示例：构建词汇表 ====================
tokens = d2l.tokenize(read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(f'词汇表大小: {len(vocab)}')
print(f'前10个高频词元: {vocab.token_freqs[:10]}')

# ==================== 示例：测试随机采样 ====================
print('\n=== 测试随机采样 ===')
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

# ==================== 示例：测试顺序分区 ====================
print('\n=== 测试顺序分区 ===')
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

# ==================== 示例：使用完整数据加载器 ====================
print('\n=== 使用完整数据加载器 ===')
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
print(f'数据集词汇表大小: {len(vocab)}')

# 验证数据迭代器
X, Y = next(iter(train_iter))
print(f'批次形状 - X: {X.shape}, Y: {Y.shape}')