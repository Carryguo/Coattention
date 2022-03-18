# 获取数据
from utils import load_data,accuracy,load_graph
from self_attention import Co_attention
import torch
import torch.optim as optim
import torch.nn.functional as F

import time
import argparse
import numpy as np

from config import Config


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')

parser.add_argument('--lr', type=float, default=0.002,
                    help='Initial learning rate.')

# default = 5e-4
parser.add_argument('--weight_decay', type=float, default=10e-4,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument("-d", "--dataset", help="dataset", type=str, default="uai")
parser.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, default=60)

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
config = Config(config_file)

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# 导入数据
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# 把数据变成[batch,n,feature]
# features = torch.unsqueeze(features,dim=1)

# 获取图结构
# sadj, fadj = load_graph(args.labelrate, config)
# feature_edges = np.genfromtxt("./data/coraml/test20.txt", dtype=np.int32)
# print(feature_edges)
sadj, fadj = load_graph(args.labelrate, config)
features, labels, idx_train, idx_test = load_data(config)

# Model and optimizer
model = Co_attention(num_attention_heads = 1,
                      input_size = features.shape[-1],
                      hidden_size1 = 300,
                      hidden_size2 = 150,
                      hidden_dropout_prob = 0,
                      class_size = labels.max().item() + 1)

# 优化器
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    # adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sadj = sadj.cuda()
    fadj = fadj.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,sadj=sadj,fadj=fadj)

    # 每次拿140个数据训练
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features,sadj=sadj,fadj=fadj)
        # output = output.view(output.shape[0], -1)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features,sadj=sadj,fadj=fadj)
    output = output.view(output.shape[0], -1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
# 训练了200次
t_total = time.time()
max_acc = 0
loss = 100
max_epoch = 0
for epoch in range(args.epochs):
    train(epoch)

    model.eval()
    output = model(features, sadj=sadj, fadj=fadj)
    output = output.view(output.shape[0], -1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    if acc_test >= max_acc:
        max_acc = acc_test
        loss = loss_test
        max_epoch = epoch


print("Test set results:",
        "loss= {:.4f}".format(loss.item()),
        "max_acc= {:.4f}".format(max_acc.item()),
         "max_epoch = {}".format(max_epoch))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()


