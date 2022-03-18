import torch
import torch.nn as nn
import math

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    # num_attention_heads is the number of attention head; input_size is the dimension of feature;
    # hidden_size is the hidden code dimension and output dimension;
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        # the dimension of every head
        self.attention_head_size = int(hidden_size / num_attention_heads)
        # the dimension of all head
        self.all_head_size = hidden_size

        # q_w q:[batch,feature Dimension]->[batch,hidden_size]
        self.query = nn.Linear(input_size, self.all_head_size)
        # k_w
        self.key = nn.Linear(input_size, self.all_head_size)
        # v_w
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

        # 全连接的判别器
        # self.result = nn.Linear(hidden_size,class_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,input_q,input_k,input_v):
        # input mm [featureDimension,hidden size]
        mixed_query_layer = self.query(input_q)
        mixed_key_layer = self.key(input_k)
        mixed_value_layer = self.value(input_v)

        # [batch,n,all_head_size]->[batch,num_attention_heads,n,attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # [batch,num_attention_heads,n,attention_head_size] mm [batch,num_attention_heads,attention_head_size,n] -> [batch,num_attention_heads,n,n]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        # 把所有头拼接起来
        # [batch,n,num_attention_heads,attention_head_size] -> [batch,n,all_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 一次全连接
        hidden_states = self.dense(context_layer)
        # hidden_states = self.out_dropout(hidden_states)
        #
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        #
        # result = self.result(hidden_states)

        return hidden_states

class Co_attention(nn.Module):
    def __init__(self,num_attention_heads,input_size,hidden_size1,hidden_size2, hidden_dropout_prob,class_size):
        super(Co_attention, self).__init__()
        #GCN
        self.gcn1 = GraphConvolution(input_size,hidden_size1)
        self.gcn2 = GraphConvolution(input_size,hidden_size1)

        #co-attention
        self.cro_att1 = SelfAttention(num_attention_heads,hidden_size1,hidden_size2,hidden_dropout_prob)
        self.cro_att2 = SelfAttention(num_attention_heads,hidden_size1,hidden_size2,hidden_dropout_prob)

        # 全连接的判别器
        self.result = nn.Linear(hidden_size2, class_size)

    def forward(self,input_feature,sadj,fadj):
        feature_g = self.gcn1(input_feature,sadj)
        feature_t = self.gcn2(input_feature,fadj)

        feature_g = torch.unsqueeze(feature_g,dim=1)
        feature_t = torch.unsqueeze(feature_t, dim=1)

        cro_att1 = self.cro_att1(input_q = feature_g,input_k = feature_t,input_v = feature_t)
        cro_att2 = self.cro_att1(input_q = feature_t,input_k = feature_g,input_v = feature_g)

        cro_att1 = cro_att1.view(cro_att1.shape[0], -1)
        cro_att2 = cro_att2.view(cro_att2.shape[0], -1)

        com = (cro_att1+cro_att2)/2

        result = self.result(com)

        return F.log_softmax(result, dim=1)





