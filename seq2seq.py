# 导入必要的库
import collections  # 用于创建默认字典，在BLEU计算中使用
import math  # 用于数学运算，特别是在BLEU计算中的指数和幂运算
import torch  # PyTorch深度学习框架
from torch import nn  # PyTorch神经网络模块
from d2l import torch as d2l  # D2L教材辅助工具库

#@save
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        """
        初始化编码器
        
        参数:
            vocab_size: 词汇表大小（源语言的词汇数量）
            embed_size: 词嵌入维度（每个词转换为多少维的向量）
            num_hiddens: GRU隐藏单元数（决定模型容量）
            num_layers: GRU层数（堆叠的循环层数量）
            dropout: Dropout概率（防止过拟合，默认0表示不使用）
        """
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层：将词索引转换为密集向量表示
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU循环神经网络：处理序列数据，输入维度是embed_size，输出隐状态维度是num_hiddens
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        """
        编码器的前向传播
        
        参数:
            X: 输入序列，形状为(batch_size, num_steps)，包含词索引
            *args: 其他参数（此处未使用，保持接口一致性）
        
        返回:
            output: GRU所有时间步的输出，形状(num_steps, batch_size, num_hiddens)
            state: GRU最终隐状态，形状(num_layers, batch_size, num_hiddens)
        """
        # 将词索引转换为词嵌入向量
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        # 将形状从(batch_size, num_steps, embed_size)转换为(num_steps, batch_size, embed_size)
        # 这是PyTorch GRU的输入要求
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        # 执行GRU前向传播
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # 包含每个时间步的输出
        # state的形状:(num_layers,batch_size,num_hiddens)
        # 包含每层的最终隐状态，用于初始化解码器
        return output, state

class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        """
        初始化解码器
        
        参数:
            vocab_size: 词汇表大小（目标语言的词汇数量）
            embed_size: 词嵌入维度
            num_hiddens: GRU隐藏单元数（应与编码器相同）
            num_layers: GRU层数（应与编码器相同）
            dropout: Dropout概率
        """
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        # 嵌入层：将目标语言的词索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU循环神经网络：输入维度是embed_size + num_hiddens（词嵌入+上下文向量拼接）
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # 全连接输出层：将隐状态映射到词汇表大小，用于预测下一个词
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        """
        初始化解码器状态
        使用编码器的最终隐状态作为解码器的初始状态
        
        参数:
            enc_outputs: 编码器输出的元组(output, state)
            *args: 其他参数（保持接口一致性）
        
        返回:
            编码器的最终隐状态，用于初始化解码器
        """
        return enc_outputs[1]

    def forward(self, X, state):
        """
        解码器的前向传播
        
        参数:
            X: 解码器输入序列，形状(batch_size, num_steps)
            state: 解码器隐状态，形状(num_layers, batch_size, num_hiddens)
        
        返回:
            output: 预测的词汇分布，形状(batch_size, num_steps, vocab_size)
            state: 更新后的解码器隐状态
        """
        # 将词索引转换为嵌入向量，然后调整维度
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        # 转换为(num_steps, batch_size, embed_size)以匹配GRU输入格式
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        # state[-1]是最顶层的隐状态，形状(batch_size, num_hiddens)
        # repeat使其在时间步维度上重复，变为(num_steps, batch_size, num_hiddens)
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 在特征维度上拼接词嵌入和上下文向量
        # 形状: (num_steps, batch_size, embed_size + num_hiddens)
        X_and_context = torch.cat((X, context), 2)
        # 通过GRU处理拼接后的输入
        output, state = self.rnn(X_and_context, state)
        # 通过全连接层得到词汇表上的分数，然后转换回(batch_size, num_steps, vocab_size)格式
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # 每个位置的输出是词汇表大小的分数向量
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

#@save
def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项（填充部分）
    用于处理变长序列，将填充位置的值设为指定值
    
    参数:
        X: 输入张量，形状(batch_size, num_steps)
        valid_len: 每个序列的有效长度，形状(batch_size,)
        value: 用于屏蔽的填充值，默认为0
    
    返回:
        屏蔽后的张量，填充位置被设为value
    """
    # 获取序列的最大长度
    maxlen = X.size(1)
    # 创建掩码：对于每个样本，标记哪些位置是有效的
    # torch.arange(maxlen)创建[0,1,2,...,maxlen-1]
    # [None, :]增加批次维度变为(1, maxlen)
    # valid_len[:, None]将(batch_size,)变为(batch_size, 1)
    # 比较后得到布尔掩码，形状(batch_size, maxlen)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # 将掩码为False的位置（即超出有效长度的位置）设为value
    X[~mask] = value
    return X

#@save
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数
    在计算损失时忽略填充位置，只对有效序列位置计算损失
    这对于变长序列非常重要，避免填充符号影响训练
    """
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        """
        计算带掩码的交叉熵损失
        
        参数:
            pred: 模型预测，形状(batch_size, num_steps, vocab_size)
            label: 真实标签，形状(batch_size, num_steps)
            valid_len: 每个序列的有效长度，形状(batch_size,)
        
        返回:
            加权后的损失，形状(batch_size,)
        """
        # 创建与标签形状相同的权重张量，初始全为1
        weights = torch.ones_like(label)
        # 使用sequence_mask将超出有效长度的位置权重设为0
        # 这样这些位置的损失将被忽略
        weights = sequence_mask(weights, valid_len)
        # 设置reduction='none'以获取每个位置的损失
        self.reduction='none'
        # 计算未加权的交叉熵损失
        # pred需要调整形状为(batch_size, vocab_size, num_steps)以匹配CrossEntropyLoss的要求
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        # 将损失与权重相乘，然后对时间步维度求平均
        # 这样只有有效位置的损失会被计入
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

#@save
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """
    训练序列到序列模型
    
    参数:
        net: 编码器-解码器模型
        data_iter: 训练数据迭代器
        lr: 学习率
        num_epochs: 训练轮数
        tgt_vocab: 目标语言词汇表
        device: 训练设备（CPU或GPU）
    """
    def xavier_init_weights(m):
        """
        Xavier均匀初始化
        用于初始化模型权重，帮助训练收敛
        
        参数:
            m: 神经网络模块
        """
        # 对线性层使用Xavier初始化
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        # 对GRU层的权重参数使用Xavier初始化
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                     xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()      # 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')

# ==================== 超参数设置 ====================
# embed_size: 词嵌入维度，将每个词映射到32维向量空间
# num_hiddens: GRU隐藏单元数，控制模型的表达能力
# num_layers: GRU堆叠层数，增加模型深度
# dropout: Dropout概率，用于正则化防止过拟合
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
# batch_size: 每批处理的样本数
# num_steps: 序列的最大长度（时间步数）
batch_size, num_steps = 64, 10
# lr: 学习率，控制参数更新的步长
# num_epochs: 训练轮数
# device: 训练设备，自动选择GPU（如果可用）否则使用CPU
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

# ==================== 数据准备 ====================
# 加载机器翻译数据集（英语-法语）
# train_iter: 训练数据迭代器
# src_vocab: 源语言（英语）词汇表
# tgt_vocab: 目标语言（法语）词汇表
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# ==================== 模型构建 ====================
# 创建编码器：将源语言序列编码为隐状态
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
# 创建解码器：基于编码器隐状态生成目标语言序列
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
# 组合编码器和解码器构成完整的Seq2Seq模型
net = d2l.EncoderDecoder(encoder, decoder)

# ==================== 模型训练 ====================
# 训练序列到序列模型
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# 显示训练过程中的损失曲线
d2l.plt.show()

#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """
    使用训练好的序列到序列模型进行预测
    将源语言句子翻译为目标语言
    
    参数:
        net: 训练好的编码器-解码器模型
        src_sentence: 源语言句子（字符串）
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        num_steps: 生成序列的最大长度
        device: 计算设备
        save_attention_weights: 是否保存注意力权重（用于可视化）
    
    返回:
        翻译后的句子（字符串）
        注意力权重序列（如果save_attention_weights=True）
    """
    # 在预测时将net设置为评估模式（关闭dropout等）
    net.eval()
    # 将源句子转换为词元索引序列
    # 先转小写，分词，然后通过词汇表转为索引，最后添加结束符<eos>
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    # 记录源序列的有效长度
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 截断或填充到固定长度num_steps
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴，从(num_steps,)变为(1, num_steps)
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # 通过编码器得到编码表示
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 使用编码器输出初始化解码器状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴，准备解码器的初始输入（开始符号<bos>）
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    # 存储输出序列和注意力权重
    output_seq, attention_weight_seq = [], []
    # 自回归生成：逐个预测下一个词
    for _ in range(num_steps):
        # 解码器前向传播，得到当前时间步的预测
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        # 贪心搜索：选择概率最大的词
        dec_X = Y.argmax(dim=2)
        # 提取预测的词索引
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论，用于注意力机制可视化）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        # 将预测的词添加到输出序列
        output_seq.append(pred)
    # 将词索引序列转换回词的字符串形式
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):  #@save
    """
    计算BLEU (Bilingual Evaluation Understudy) 分数
    BLEU是评估机器翻译质量的标准指标
    通过比较n-gram匹配来衡量预测序列与参考序列的相似度
    
    参数:
        pred_seq: 预测的序列（字符串）
        label_seq: 参考序列/标签（字符串）
        k: 计算到k-gram的匹配（通常k=4）
    
    返回:
        BLEU分数，范围[0, 1]，越高越好
    """
    # 将序列分割为词列表
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    # 获取预测序列和标签序列的长度
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # 计算长度惩罚项（brevity penalty）
    # 如果预测序列比参考序列短，会受到惩罚
    score = math.exp(min(0, 1 - len_label / len_pred))
    # 计算1-gram到k-gram的精确度
    for n in range(1, k + 1):
        # num_matches: n-gram匹配数量
        # label_subs: 存储参考序列中n-gram的出现次数
        num_matches, label_subs = 0, collections.defaultdict(int)
        # 统计参考序列中所有n-gram及其出现次数
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        # 遍历预测序列的n-gram，检查是否在参考序列中
        for i in range(len_pred - n + 1):
            # 如果当前n-gram在参考序列中存在
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                # 减少计数，确保每个n-gram只匹配一次
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        # 计算n-gram精确度并累乘到分数中
        # 使用几何平均（权重为0.5^n）
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

# ==================== 模型测试与评估 ====================
# 准备测试用的英语-法语句子对
# engs: 英语源句子
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# fras: 对应的法语参考翻译
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# 对每个测试句子进行翻译并计算BLEU分数
for eng, fra in zip(engs, fras):
    # 使用模型进行预测翻译
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    # 打印源句子、翻译结果和BLEU分数（k=2表示计算到bi-gram）
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')