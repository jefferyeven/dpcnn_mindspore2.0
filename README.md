dpcnn模型自验报告
（梁瑞、1249336792@qq.com）

1.	模型简介
1.1.	网络模型结构简介：
Dpcnn
1.2.	数据集：
www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
1.3.	代码提交地址：https://github.com/jefferyeven/dpcnn_mindspore2.0
2.	代码目录结构说明
 
3.	自验结果
3.1.	自验环境：
（所用硬件环境、MindSpore版本、python第三方库等说明）
3.2.	训练超参数：
4.	def __init__(self):
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
4.1.	训练：
模型训练
4.1.1.	如何启动训练脚本：
运行 data_helpers.py
运行 train.py
4.1.2.	训练精度结果：
 
