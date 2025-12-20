"""文本预处理模块

本模块实现了自然语言处理中的文本预处理功能, 包括:
1. 文本数据读取与清洗
2. 分词(Tokenization): 将文本拆分为单词或字符
3. 词表(Vocabulary)构建: 建立词元到索引的映射
4. 语料库(Corpus)生成: 将文本转换为索引序列

主要用于序列模型(如RNN, LSTM)的数据准备阶段.
"""

import collections  # 用于统计词频(Counter)
import re  # 正则表达式, 用于文本清洗
from d2l import torch as d2l  # D2L工具库

# 注册时间机器数据集到D2L数据中心
# DATA_HUB: 数据集名称 -> (下载URL, SHA-1哈希值)
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中
    
    数据清洗步骤:
    1. 下载并读取时间机器文本文件
    2. 使用正则表达式将非字母字符替换为空格
    3. 去除首尾空白并转换为小写
    
    Returns:
        list[str]: 清洗后的文本行列表
    """
    # 下载数据集并打开文件
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 文本规范化: 只保留字母, 其他字符替换为空格, 转小写
    # re.sub('[^A-Za-z]+', ' ', line): 将非字母字符替换为空格
    # strip(): 去除首尾空白
    # lower(): 转换为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# 读取时间机器数据集
lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])  # 打印第一行文本(索引0)
print(lines[10])  # 打印第11行文本(索引10)

def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元
    
    分词是NLP的基础步骤, 将文本切分为最小的语义单元.
    
    Args:
        lines (list[str]): 文本行列表
        token (str): 词元类型, 'word'表示按单词分词, 'char'表示按字符分词
        
    Returns:
        list[list[str]]: 二维列表, 每个子列表是一行文本的词元序列
    """
    if token == 'word':
        # 按空格分割, 得到单词级别的词元
        return [line.split() for line in lines]
    elif token == 'char':
        # 将每行文本转换为字符列表, 得到字符级别的词元
        return [list(line) for line in lines]
    else:
        print('错误: 未知词元类型: ' + token)

# 对文本进行分词(默认按单词)
tokens = tokenize(lines)
# 打印前11行的分词结果
for i in range(11):
    print(tokens[i])

class Vocab:  #@save
    """文本词表类
    
    词表(Vocabulary)是词元到整数索引的映射, 用于将文本转换为数值表示.
    核心功能:
    1. 统计词频并排序
    2. 过滤低频词(min_freq)
    3. 建立双向映射: 词元<->索引
    4. 处理未知词元(<unk>)
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """初始化词表
        
        Args:
            tokens (list): 词元列表(可以是1D或2D列表)
            min_freq (int): 最小词频阈值, 低于此频率的词元会被过滤
            reserved_tokens (list): 保留词元列表(如<pad>, <bos>, <eos>等特殊标记)
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计所有词元的出现频率
        counter = count_corpus(tokens)
        # 按词频降序排序, 得到(词元, 频率)元组列表
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 初始化词元列表: <unk>(索引0) + 保留词元 + 高频词元
        # <unk>: unknown token, 用于表示词表中不存在的词
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 构建词元到索引的映射字典
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        # 将满足最小频率要求的词元加入词表
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break  # 因为已按频率降序排列, 后续词元频率更低
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """返回词表大小(唯一词元数量)"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """将词元转换为索引
        
        支持单个词元或词元列表的转换.
        未知词元会被映射到<unk>的索引(0).
        
        Args:
            tokens: 单个词元(str)或词元列表(list/tuple)
            
        Returns:
            int或list[int]: 对应的索引或索引列表
        """
        if not isinstance(tokens, (list, tuple)):
            # 单个词元: 返回其索引, 如果不存在则返回<unk>的索引
            return self.token_to_idx.get(tokens, self.unk)
        # 词元列表: 递归调用, 返回索引列表
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """将索引转换回词元(逆操作)
        
        Args:
            indices: 单个索引(int)或索引列表(list/tuple)
            
        Returns:
            str或list[str]: 对应的词元或词元列表
        """
        if not isinstance(indices, (list, tuple)):
            # 单个索引: 直接返回对应词元
            return self.idx_to_token[indices]
        # 索引列表: 返回对应的词元列表
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """返回未知词元<unk>的索引(固定为0)"""
        return 0

    @property
    def token_freqs(self):
        """返回词元频率列表: [(词元, 频率), ...], 按频率降序排列"""
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率
    
    使用Counter对象高效统计每个词元的出现次数.
    
    Args:
        tokens: 词元列表, 可以是:
               - 1D列表: [token1, token2, ...]
               - 2D列表: [[token1, token2], [token3, token4], ...]
               
    Returns:
        collections.Counter: 词元计数器, {词元: 频率}
    """
    # 如果是2D列表(每行是一个词元列表), 需要先展平
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 列表推导式展平: 逐行遍历, 逐词元提取
        tokens = [token for line in tokens for token in line]
    # 使用Counter统计每个词元的出现次数
    return collections.Counter(tokens)

# 构建词表(使用分词后的tokens)
vocab = Vocab(tokens)
# 打印词表中前10个词元及其索引
print(list(vocab.token_to_idx.items())[:10])

# 演示词元到索引的转换
for i in [0, 10]:
    print('文本:', tokens[i])  # 原始词元列表
    print('索引:', vocab[tokens[i]])  # 转换为索引列表

def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表
    
    这是一个完整的数据预处理流水线, 整合了:
    读取 -> 清洗 -> 分词 -> 建立词表 -> 转换为索引序列
    
    Args:
        max_tokens (int): 最大词元数量限制, -1表示不限制
        
    Returns:
        tuple: (corpus, vocab)
            - corpus (list[int]): 词元索引的一维列表
            - vocab (Vocab): 词表对象
    """
    # 步骤1: 读取并清洗文本
    lines = read_time_machine()
    # 步骤2: 字符级分词(每个字符作为一个词元)
    tokens = tokenize(lines, 'char')
    # 步骤3: 构建词表
    vocab = Vocab(tokens)
    # 步骤4: 将所有文本展平为一维索引序列
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落,
    # 所以将所有文本行展平到一个列表中, 形成连续的字符流
    corpus = [vocab[token] for line in tokens for token in line]
    # 如果指定了最大词元数, 则截断
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# 加载完整的时光机器语料库(字符级)
corpus, vocab = load_corpus_time_machine()
# 打印语料库长度(词元总数)和词表大小(唯一字符数)
print(len(corpus), len(vocab))