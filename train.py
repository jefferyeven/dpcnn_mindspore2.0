from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, Callback, LossMonitor, SummaryCollector, \
    TimeMonitor
from utils import *
import argparse
import mindspore.dataset as ds
from models import *
import mindspore as ms
import mindspore.nn as nn
from mindspore import Model, context
from mindspore import load_checkpoint, load_param_into_net
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore.nn.metrics import Accuracy
import model

# callback
class EvalCheckpoint(Callback):

    def __init__(self, model, net, dev_dataset, steps):
        self.model = model
        self.net = net
        self.dev_dataset = dev_dataset
        self.steps = steps
        self.counter = 0
        self.best_acc = 0

    def step_end(self, run_context):
        self.counter = self.counter + 1
        if self.counter % self.steps == 0:
            self.counter = 0
            acc = self.model.eval(self.dev_dataset, dataset_sink_mode=False)
            if acc['Accuracy'] > self.best_acc:
                self.best_acc = acc['Accuracy']
                mindspore.save_checkpoint(self.net, 'dpcnn.ckpt')
            print("{}".format(acc))
if __name__ =='__main__':
    config = model.SetConfig()
    np.random.seed(1)
    start_time = time.time()
    print("Loading data......")

     # load data
    vocab, train_data, test_data = build_dataset(config, True)

    # train
    train_iter = DatasetMSIterater(train_data, config)
    train_data = ds.GeneratorDataset(train_iter, ['data', 'label'], num_parallel_workers=1, shuffle=True)
    train_dataset = train_data.map(C.TypeCast(mstype.int32), input_columns='label', num_parallel_workers=1)
    # test
    test_iter = DatasetMSIterater(test_data, config)
    test_data = ds.GeneratorDataset(test_iter, ['data', 'label'], num_parallel_workers=1, shuffle=True)
    test_dataset = test_data.map(C.TypeCast(mstype.int32), input_columns='label', num_parallel_workers=1)

# init net
    config.n_vocab = len(vocab)
    net = DPCNN(config)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    optim = nn.Adam(net.trainable_params(), learning_rate=config.learning_rate)
    model = ms.Model(net, loss_fn = loss, optimizer = optim, metrics = {"Accuracy": Accuracy()})
    model.train(epoch=20, train_dataset=train_dataset,
                callbacks=[LossMonitor(10), EvalCheckpoint(model, net, test_dataset, 10)])

    param_dict = mindspore.load_checkpoint('dpcnn.ckpt')
    mindspore.load_param_into_net(net, param_dict)
    model = Model(net, loss_fn=loss, optimizer=None, metrics={"Accuracy": Accuracy()})
    acc = model.eval(test_dataset, dataset_sink_mode=False)
    print("{}".format(acc))



