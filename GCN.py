import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy as np


'''
Standard GCN: 
1、 aggragate neighbor nodes with "degree weights"
2、 apply linear transformation and activation function on new node features.
'''


'''
custom message passing functions, default as follows.
'''
# Easiest 
# gcn_msg = fn.copy_src(src = 'h', out = 'm')
# gcn_reduce = fn.mean(msg = 'm', out = 'h')


# Standard
# need assgin degree features to nodes first: g.ndata['deg'] = g.out_degrees(g.nodes()).float()

def message_func(edges):
    #print(edges.src['h'].size())
    return {'m': (1 / (np.sqrt(edges.src['deg']) * np.sqrt(edges.dst['deg']))).view(-1,1) * edges.src['h']}

def reduce_func(nodes):
    msgs = torch.cat((nodes.mailbox['m'], nodes.data['h'].unsqueeze(1)), dim = 1)   # mailbox size (batch * neighbors * feats), nodes.data['h'] size (batch * feats), so they need to be concatenated at dim 1. 
    msgs = torch.mean(msgs, dim = 1)    
    return {'h': msgs}


'''
linear transformation, can be written as a Node UDF, then call apply_nodes() to execute. Surely, it can combine with reduce_func. see ReduceLayer
'''

class NodeTransLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(NodeTransLayer, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
    
    def forward(self, node):
        h = self.fc(node.data['h'])
        h = F.relu(h)
        return {'h': h}


class ReduceLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(NodeTransLayer, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
    
    def forward(self, nodes):
        h = torch.cat((nodes.mailbox['m'], nodes.data['h'].unsqueeze(1)), dim = 1)      
        h = torch.mean(h, dim = 1)
        h = F.relu(self.fc(h))
        return {'h': h}

'''
GCN 
'''
class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.node_trans = NodeTransLayer(in_feats, out_feats)

    def forward(self, g, features):
        g.ndata['h'] = features
        g.update_all(message_func, reduce_func)
        g.apply_nodes(func = self.node_trans)       # this step can be combined in message passing step.
        return g.ndata.pop('h')




# A two-layer GCN
class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.gcn1 = GCN(21, 21)
        self.gcn2 = GCN(21, 21)
        # self.fc = nn.Linear(70, 15)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        g.ndata['h'] = x
        # hg = dgl.mean_nodes(g, 'h')
        return x
