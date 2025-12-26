"""D2L 深度学习项目 - 通用工具函数集合

本模块整合了《动手学深度学习》项目中所有标记 #@save 的可复用函数。
包括数据处理、模型训练、评估指标、可视化等工具。

使用方式:
    from d2l_utils import synthetic_data, sgd, read_time_machine
"""

import collections
import hashlib
import os
import random
import re
import tarfile
import zipfile

import requests
import torch
from torch import nn
from torch.nn import functional as F
from IPython import display
from d2l import torch as d2l


# ==================== 数据集注册与下载 ====================

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# 注册时间机器数据集
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',
    '090b5e7e70c295757f55df93cb0a180b9691891a'
)

# 注册Kaggle房价预测数据集
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce'
)

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
)


def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名
    
    Args:
        name: 数据集名称(在DATA_HUB中注册的键名)
        cache_dir: 缓存目录路径
        
    Returns:
        str: 下载后的本地文件路径
    """
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    
    # 检查文件是否已存在且哈希值正确
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    
    # 下载文件
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件
    
    Args:
        name: 数据集名称
        folder: 解压后的文件夹名称
        
    Returns:
        str: 解压后的目录路径
    """
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


# ==================== 文本数据处理 ====================

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中
    
    数据清洗步骤:
    1. 下载并读取时间机器文本文件
    2. 使用正则表达式将非字母字符替换为空格
    3. 去除首尾空白并转换为小写
    
    Returns:
        list[str]: 清洗后的文本行列表
    """
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):  #@save
    """将文本行列表拆分为单词或字符词元
    
    Args:
        lines (list[str]): 文本行列表
        token (str): 分词类型, 'word'按单词分词, 'char'按字符分词
        
    Returns:
        list[list[str]]: 二维列表, 每行是一个词元列表
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def count_corpus(tokens):  #@save
    """统计词元的频率
    
    Args:
        tokens: 词元列表, 可以是1D列表或2D列表
               
    Returns:
        collections.Counter: 词元计数器, {词元: 频率}
    """
    # 如果是2D列表(每行是一个词元列表), 需要先展平
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:  #@save
    """文本词表
    
    将词元映射到数字索引, 并记录每个词元的频率.
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """初始化词表
        
        Args:
            tokens (list): 词元列表(可以是1D或2D列表)
            min_freq (int): 最小词频阈值
            reserved_tokens (list): 保留词元列表(如<pad>, <bos>, <eos>)
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        # 统计词频并按降序排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        
        # 未知词元索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx 
                            for idx, token in enumerate(self.idx_to_token)}
        
        # 添加满足最小频率要求的词元
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs


def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表
    
    Args:
        max_tokens (int): 返回的最大词元数, -1表示不限制
        
    Returns:
        tuple: (corpus, vocab)
            - corpus: 词元索引列表
            - vocab: 词表对象
    """
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 将所有文本行展平成一个列表
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# ==================== 序列数据迭代器 ====================

def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列
    
    Args:
        corpus: 词元索引列表
        batch_size: 批量大小
        num_steps: 每个子序列的时间步数
        
    Yields:
        tuple: (X, Y) 特征和标签张量
    """
    # 从随机偏移量开始对序列进行分区
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1是因为需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 随机打乱
    random.shuffle(initial_indices)
    
    def data(pos):
        return corpus[pos: pos + num_steps]
    
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列
    
    Args:
        corpus: 词元索引列表
        batch_size: 批量大小
        num_steps: 每个子序列的时间步数
        
    Yields:
        tuple: (X, Y) 特征和标签张量
    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表
    
    Args:
        batch_size: 批量大小
        num_steps: 时间步数
        use_random_iter: 是否使用随机采样
        max_tokens: 最大词元数
        
    Returns:
        tuple: (data_iter, vocab)
    """
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# ==================== 线性回归相关 ====================

def synthetic_data(w, b, num_examples):  #@save
    """生成线性回归合成数据: y = Xw + b + 噪声
    
    Args:
        w: 真实权重向量
        b: 真实偏置
        num_examples: 样本数量
        
    Returns:
        tuple: (X, y) 特征矩阵和标签向量
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def linreg(X, w, b):  #@save
    """线性回归模型
    
    Args:
        X: 输入特征
        w: 权重
        b: 偏置
        
    Returns:
        预测值
    """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):  #@save
    """均方损失函数
    
    Args:
        y_hat: 预测值
        y: 真实值
        
    Returns:
        损失值
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降优化器
    
    Args:
        params: 参数列表
        lr: 学习率
        batch_size: 批量大小
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# ==================== 图像分类相关 ====================

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签
    
    Args:
        labels: 标签索引列表
        
    Returns:
        list[str]: 文本标签列表
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def get_dataloader_workers():  #@save
    """使用4个进程来读取数据
    
    Returns:
        int: 工作进程数
    """
    return 4


def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中
    
    Args:
        batch_size: 批量大小
        resize: 调整图像大小
        
    Returns:
        tuple: (train_iter, test_iter) 训练和测试数据迭代器
    """
    from torchvision import transforms
    import torchvision
    from torch.utils import data
    
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                           num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                           num_workers=get_dataloader_workers()))


class Accumulator:  #@save
    """在n个变量上累加
    
    用于累积训练过程中的损失和准确率等指标.
    """
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def accuracy(y_hat, y):  #@save
    """计算预测正确的数量
    
    Args:
        y_hat: 预测值
        y: 真实标签
        
    Returns:
        float: 正确预测的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度
    
    Args:
        net: 神经网络模型
        data_iter: 数据迭代器
        
    Returns:
        float: 准确率
    """
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）
    
    Args:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        updater: 优化器
        
    Returns:
        tuple: (训练损失, 训练准确率)
    """
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）
    
    Args:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        loss: 损失函数
        num_epochs: 训练轮数
        updater: 优化器
    """
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# ==================== GPU训练相关 ====================

def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """使用GPU计算模型在数据集上的精度
    
    Args:
        net: 神经网络模型
        data_iter: 数据迭代器
        device: 计算设备
        
    Returns:
        float: 准确率
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):  #@save
    """用GPU训练模型(在第六章定义)
    
    Args:
        net: 神经网络模型
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        num_epochs: 训练轮数
        lr: 学习率
        device: 计算设备
    """
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# ==================== 其他工具函数 ====================

def dropout_layer(X, dropout):  #@save
    """Dropout 层实现
    
    Args:
        X: 输入张量
        dropout: dropout概率
        
    Returns:
        应用dropout后的张量
    """
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失
    
    Args:
        net: 神经网络模型
        data_iter: 数据迭代器
        loss: 损失函数
        
    Returns:
        float: 平均损失
    """
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


# ==================== 残差网络相关 ====================

class Residual(nn.Module):  #@save
    """残差块 (Residual Block)
    
    实现公式: F(X) = ReLU(Conv2(ReLU(BN(Conv1(X)))) + X)
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, X):
        Y = torch.nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return torch.nn.functional.relu(Y)


# ==================== RNN模型相关 ====================

class RNNModel(nn.Module):  #@save
    """循环神经网络模型（使用PyTorch内置RNN层）
    
    这是一个通用的RNN模型封装类，支持RNN、LSTM、GRU等不同类型的循环层。
    适用于序列建模任务，如语言模型、文本分类等。
    """
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        """初始化RNN模型
        
        Args:
            rnn_layer: PyTorch的RNN层（nn.RNN、nn.LSTM或nn.GRU）
            vocab_size: 词汇表大小（输出维度）
            **kwargs: 传递给nn.Module的其他参数
        """
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        
        # 判断是否为双向RNN
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        """前向传播
        
        Args:
            inputs: 输入序列，形状为(batch_size, num_steps)
            state: 隐藏状态
            
        Returns:
            tuple: (output, state)
                - output: 输出序列，形状为(num_steps*batch_size, vocab_size)
                - state: 更新后的隐藏状态
        """
        # 将输入索引转换为one-hot编码
        # inputs.T: (batch_size, num_steps) -> (num_steps, batch_size)
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        
        # RNN前向传播
        Y, state = self.rnn(X, state)
        
        # 全连接层首先将Y的形状改为(时间步数*批量大小, 隐藏单元数)
        # 它的输出形状是(时间步数*批量大小, 词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        """初始化隐藏状态
        
        Args:
            device: 计算设备
            batch_size: 批量大小
            
        Returns:
            初始隐藏状态（零向量）
                - 对于RNN/GRU: 返回张量
                - 对于LSTM: 返回元组(h, c)
        """
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                               batch_size, self.num_hiddens),
                              device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


class RNNModelScratch:  #@save
    """从零开始实现的循环神经网络模型
    
    这是一个通用的 RNN 模型封装类，支持不同的 RNN 变体（vanilla RNN, LSTM, GRU）。
    通过传入不同的 forward_fn 和 init_state 函数来实现不同类型的 RNN。
    """
    
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """初始化 RNN 模型
        
        Args:
            vocab_size: 词表大小
            num_hiddens: 隐藏单元数量
            device: 计算设备
            get_params: 参数初始化函数
            init_state: 状态初始化函数
            forward_fn: 前向传播函数（rnn/lstm/gru）
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 初始化模型参数
        self.params = get_params(vocab_size, num_hiddens, device)
        # 保存初始化和前向传播函数的引用
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """前向传播（使模型可以像函数一样调用）
        
        Args:
            X: 输入索引，形状 (批量大小, 时间步数)
            state: 隐藏状态
            
        Returns:
            tuple: (输出, 新状态)
        """
        # 将输入索引转换为独热编码
        # X.T: (时间步数, 批量大小)
        # one_hot 后: (时间步数, 批量大小, 词表大小)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """返回初始隐藏状态
        
        Args:
            batch_size: 批量大小
            device: 计算设备
            
        Returns:
            初始隐藏状态（通常是零向量）
        """
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """基于前缀生成新字符（文本生成函数）
    
    使用训练好的 RNN 模型进行文本生成。
    过程分为两个阶段：
    1. 预热期：用 prefix 中的字符更新隐藏状态
    2. 生成期：根据当前状态预测下一个字符，并继续生成
    
    Args:
        prefix: 前缀字符串，用于"预热"模型
        num_preds: 要生成的新字符数量
        net: RNN 模型
        vocab: 词表对象
        device: 计算设备
        
    Returns:
        str: 包含前缀和生成字符的完整字符串
    """
    # 初始化隐藏状态（批量大小为 1，因为只生成一个序列）
    state = net.begin_state(batch_size=1, device=device)
    
    # outputs 存储生成的字符索引，从前缀的第一个字符开始
    outputs = [vocab[prefix[0]]]
    
    # 辅助函数：将最后一个输出索引转换为模型输入格式
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    # 预热期：用前缀中的字符逐步更新隐藏状态
    # 这样模型能够"理解"前缀的上下文信息
    for y in prefix[1:]:  
        _, state = net(get_input(), state)  # 更新状态但不使用输出
        outputs.append(vocab[y])  # 将前缀字符添加到输出
    
    # 生成期：预测接下来的 num_preds 个字符
    for _ in range(num_preds):  
        y, state = net(get_input(), state)  # 前向传播获得预测
        # 选择概率最高的字符（贪心解码）
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    
    # 将索引序列转换回字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):  #@save
    """梯度裁剪（解决 RNN 训练中的梯度爆炸问题）
    
    RNN 在训练过程中容易出现梯度爆炸（梯度值变得非常大）。
    梯度裁剪通过限制梯度的 L2 范数来缓解这个问题。
    
    原理：
        如果 ||g|| > θ，则 g = θ * g / ||g||
        即：将梯度缩放到最大范数为 θ
    
    Args:
        net: 神经网络模型
        theta: 梯度裁剪阈值
    """
    # 获取所有需要梯度的参数
    if isinstance(net, nn.Module):
        # PyTorch 内置模型
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        # 自定义模型（如 RNNModelScratch）
        params = net.params
    
    # 计算所有参数梯度的 L2 范数：||g|| = sqrt(sum(g_i^2))
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    
    # 如果梯度范数超过阈值，进行裁剪
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm  # 缩放梯度


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):  #@save
    """训练RNN网络一个迭代周期
    
    完成一个 epoch 的训练，处理所有批次的数据。
    
    Args:
        net: RNN 模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        updater: 优化器或自定义更新函数
        device: 计算设备
        use_random_iter: 是否使用随机采样（影响隐藏状态的处理）
        
    Returns:
        tuple: (困惑度, 训练速度)
            - 困惑度 = exp(平均损失)，衡量模型预测的不确定性
            - 训练速度：每秒处理的词元数量
    """
    import math
    state, timer = None, d2l.Timer()  # 初始化状态和计时器
    metric = d2l.Accumulator(2)  # 累加器：[总损失, 词元总数]
    
    # 遍历所有批次
    for X, Y in train_iter:
        # 隐藏状态的初始化和分离
        if state is None or use_random_iter:
            # 情况1：第一次迭代 或 使用随机采样
            # 随机采样时每个批次是独立的，需要重新初始化状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 情况2：使用顺序采样
            # 需要从计算图中分离隐藏状态，避免反向传播到太久远的时间步
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 针对 nn.GRU：state 是张量
                state.detach_()
            else:
                # 针对 nn.LSTM 或自定义模型：state 是元组
                for s in state:
                    s.detach_()
        
        # 准备标签：转置并展平
        # Y: (batch_size, num_steps) -> Y.T: (num_steps, batch_size) 
        # -> reshape(-1): (batch_size * num_steps,)
        y = Y.T.reshape(-1)
        
        # 将数据移到指定设备
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        y_hat, state = net(X, state)
        
        # 计算损失（交叉熵）
        l = loss(y_hat, y.long()).mean()
        
        # 反向传播和参数更新
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 优化器
            updater.zero_grad()      # 梯度清零
            l.backward()             # 反向传播
            grad_clipping(net, 1)    # 梯度裁剪
            updater.step()           # 更新参数
        else:
            # 使用自定义的 SGD 更新函数
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)    # batch_size=1 因为损失已经取了平均
        
        # 累加损失和词元数
        metric.add(l * y.numel(), y.numel())
    
    # 计算困惑度和训练速度
    # 困惑度 = exp(平均损失)，越低越好
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    """RNN 模型完整训练流程
    
    训练 RNN 模型，并定期显示生成的文本和困惑度曲线。
    
    Args:
        net: RNN 模型
        train_iter: 训练数据迭代器
        vocab: 词表对象
        lr: 学习率
        num_epochs: 训练轮数
        device: 计算设备
        use_random_iter: 是否使用随机采样（默认 False，使用顺序采样）
    """
    # 定义交叉熵损失函数（用于分类任务）
    loss = nn.CrossEntropyLoss()
    
    # 创建动画器，用于实时绘制训练曲线
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    
    # 初始化优化器
    if isinstance(net, nn.Module):
        # PyTorch 内置模型：使用 SGD 优化器
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 自定义模型：使用自定义 SGD 函数
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    
    # 定义预测函数（生成 50 个字符）
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练一个 epoch
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        
        # 每 10 个 epoch 打印生成的文本并更新图表
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))  # 展示生成效果
            animator.add(epoch + 1, [ppl])    # 添加困惑度数据点
    
    # 训练结束，打印最终统计信息
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    # 展示最终的文本生成效果
    print(predict('time traveller'))
    print(predict('traveller'))
