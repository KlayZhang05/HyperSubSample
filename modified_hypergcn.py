#!/usr/bin/env python3
"""
真正的HyperGCN实现 - 基于原项目
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
from tqdm import tqdm


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


class HyperGraphConvolution(nn.Module):
    """
    HyperGCN卷积层 - 与原项目完全一致
    """

    def __init__(self, a, b, reapproximate=False, cuda=True):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, m=True):
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            print("HyperGraphConvolution\n")
            A = Laplacian(n, structure, X, m)
        else: 
            A = structure

        if self.cuda: 
            A = A.cuda()
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b


def Laplacian(V, E, X, m):
    """
    真正的Laplacian近似 - 与原项目完全一致
    """
    print("Laplacian了\n")
    edges, weights = [], {}
    rv = np.random.rand(X.shape[1])
# tqdm(range(self.config.epochs), desc="训练进度")
    for k in tqdm(E.keys(),desc="近似进度"):
        hyperedge = list(E[k])
        
        p = np.dot(X[hyperedge], rv)   # projection onto a random vector rv
        s, i = np.argmax(p), np.argmin(p)
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators
        c = 2*len(hyperedge) - 3    # normalisation constant
        if m:
            # connect the supremum (Se) with the infimum (Ie)
            edges.extend([[Se, Ie], [Ie, Se]])
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)
            
            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
                    weights = update(Se, Ie, mediator, weights, c)
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            e = len(hyperedge)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)    
    
    return adjacency(edges, weights, V)


def update(Se, Ie, mediator, weights, c):
    """更新边权重"""
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights


def adjacency(edges, weights, n):
    """构建邻接矩阵"""
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A


def symnormalise(M):
    """对称归一化"""
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 


def ssm2tst(M):
    """稀疏矩阵转torch张量"""
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)


class ModifiedHyperGCN(nn.Module):
    """
    真正的HyperGCN - 与原项目架构一致
    """
    
    def __init__(self, V, E, X, embedding_dim=64, depth=2, dropout=0.3, 
                 mediators=True, fast=True, cuda=True):
        super(ModifiedHyperGCN, self).__init__()
        
        d, l = X.shape[1], depth
        self.cuda_enabled = cuda and torch.cuda.is_available()

        # 构建层次结构 - 与原项目一致
        h = [d]
        for i in range(l-1):
            power = l - i + 2
            h.append(2**power)
        h.append(embedding_dim)

        if fast:
            print("fast\n")
            reapproximate = False
            structure = Laplacian(V, E, X.cpu().detach().numpy(), mediators)        
        else:
            reapproximate = True
            structure = E
            
        self.layers = nn.ModuleList([
            HyperGraphConvolution(h[i], h[i+1], reapproximate, self.cuda_enabled) 
            for i in range(l)
        ])
        
        self.do, self.l = dropout, depth
        self.structure, self.m = structure, mediators
        
        # 存储节点特征
        self.node_features = X
        if self.cuda_enabled:
            self.node_features = self.node_features.cuda()

    def forward(self, H):
        """
        前向传播 - 与原项目完全一致
        """
        do, l, m = self.do, self.l, self.m
        
        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                H = F.dropout(H, do, training=self.training)
        
        return H
    
    def get_node_embeddings(self):
        """获取所有节点的嵌入"""
        with torch.no_grad():
            return self.forward(self.node_features)

# 测试代码
if __name__ == "__main__":
    print("=== ModifiedHyperGCN测试 ===")
    try:
        # 测试数据
        num_nodes = 50
        feature_dim = 16
        X = torch.randn(num_nodes, feature_dim)
        E = {0: [0, 1, 2], 1: [3, 4, 5]}
        
        # 创建模型（使用CPU避免CUDA问题）
        model = ModifiedHyperGCN(num_nodes, E, X, embedding_dim=32, cuda=False)
        
        # 测试
        embeddings = model.get_node_embeddings()
        print(f"✓ 成功！嵌入形状: {embeddings.shape}")
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
