import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    # 转化成元组
    classes = set(labels)
    # c为类别, i是第i+1个类别,后面np.identity(len(classes))[i, :]是one_hot, np.identity其实是生成一个对角线为1, 其他元素为0的矩阵, 一共有七类
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # 按照labels收到的类别进行one_hot编码,例:array([0., 0., 0., 0., 1., 0., 0.]), array([0., 0., 1., 0., 0., 0., 0.]), array([0., 1., 0., 0., 0., 0., 0.])
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    # 返回one_hot编码
    return labels_onehot


# def load_data(path="data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#     # 传入content文件数据, 格式为(2708, 1435) 例:['31336' '0' '0' ... '0' '0' 'Neural_Networks']
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
#                                         dtype=np.dtype(str))
#
#     # 显示出稀疏矩阵中,所有元素的位置和元素值,因为输入是str格式,所以可以输出全部位置
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#
#     # 把content里面, 最后那个类型进行编码
#     labels = encode_onehot(idx_features_labels[:, -1])
#
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)# 把文章的编号搞成列表
#     # 把文章的编号做成字典
#     idx_map = {j: i for i, j in enumerate(idx)}
#
#     # 传入cites文件的文章引用数据,例: [[     35    1033]
#     #  [     35  103482]
#     #  [     35  103515]]
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
#                                     dtype=np.int32)
#     # flatten()是把矩阵拍成一维的数组, 然后获取编号的顺序，然后变回edges_unordered原来的形状
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#
#     # 创建一个文章索引的邻接矩阵,相当于一个有向图, 例: adj:   (163, 402)	1.0
#     #   (163, 659)	1.0
#     #   (163, 1696)	1.0
#     #   (163, 2295)	1.0
#     #   (163, 1274)	1.0
#     #   (163, 1286)	1.0
#     #   (163, 1544)	1.0
#     #   (163, 2600)	1.0
#     #   (163, 2363)	1.0
#     #   (163, 1905)	1.0
#     #   (163, 1611)	1.0
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)
#
#     # build symmetric adjacency matrix
#     # 相当于把有向图做成一个无向图,对称矩阵
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     # 特征矩阵行归一化
#     features = normalize(features)
#
#     # 这是Abar 加行归一化
#     adj = normalize(adj + sp.eye(adj.shape[0]))
#     # 训练
#     idx_train = range(140)
#     # 验证
#     idx_val = range(200, 500)
#     # 测试
#     idx_test = range(500, 1500)
#     # 变为正常的矩阵, 之前那个是三元组 也就是 (raw,col) data
#     features = torch.FloatTensor(np.array(features.todense()))
#     # 变成0 1 2 3 4
#     labels = torch.LongTensor(np.where(labels)[1])
#
#     # 把scipy.sparse转换成torch的sparse矩阵
#     adj = sparse_mx_to_torch_sparse_tensor(adj)
#
#     idx_train = torch.LongTensor(idx_train)
#
#     idx_val = torch.LongTensor(idx_val)
#
#     idx_test = torch.LongTensor(idx_test)
#
#     return adj, features, labels, idx_train, idx_val, idx_test



def normalize(mx):
    """Row-normalize sparse matrix"""
    # 行元素相加,比如[[0, 1, 0], [0, 0, 0], [1, 0, 0],[1,0,0]] 加完之后等于[1 0 1]
    rowsum = np.array(mx.sum(1))

    r_inv = np.power(rowsum, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.

    r_mat_inv = sp.diags(r_inv)
    # 左乘
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    # 获取每行数据的最大值和数据转换
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data(config):
    f = np.loadtxt(config.feature_path, dtype = float)
    l = np.loadtxt(config.label_path, dtype = int)
    test = np.loadtxt(config.test_path, dtype = int)
    train = np.loadtxt(config.train_path, dtype = int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_graph(dataset, config):

    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj, nfadj