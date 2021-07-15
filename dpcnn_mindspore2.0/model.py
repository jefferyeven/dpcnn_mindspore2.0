import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops
import mindspore

class SetConfig():
    """配置参数"""
    def __init__(self):
        self.model_name = 'DPCNN'
        self.train_path = './data/train.txt'  # 训练集
        self.test_path = './data/test.txt'#测试集
        self.class_list = [x.strip() for x in open(
                           './data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path ='./data/vocab.pkl'                                # 词表
        self.dropout = 0.5
        self.kernels = 250                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                               # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embedding_pretrained = None
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.num_filters = 250                                          # 卷积核数量(channels数)


class DPCNN(nn.Cell):
    def __init__( self, config):
        super(DPCNN, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embed), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.relu = nn.ReLU()
        self.padding1 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (0, 0)), mode='CONSTANT')
        self.padding2 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 0)), mode='CONSTANT')
        self.fc = nn.Dense(self.kernels, self.num_classes)

    def construct(self, x):
        x = self.embedding(x)
        x = ops.ExpandDims()(x, 1)       # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)     # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)         # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)     # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)         # [batch_size, 250, seq_len-3+1, 1]
        while x.shape[-2] > 2:
            x = self._block(x)
        x = x.squeeze()          # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)
        x = self.padding1(px)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        # Short Cut
        x = x + px
        return x

