from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
import math

from numpy import random
import numpy as np
import itertools

class LsrHierGNNLayer(nn.Module):
    def __init__(self, args):
        super(LsrHierGNNLayer, self).__init__()
        self.args = args
        self.hop_layers = self.args.hop_layer
        self.matrix_tree_layer = Matrix_Tree_Layer(self.args)
        self.msg_psg_layer = nn.ModuleList([Message_Passing_Layer(self.args) for i in range(self.hop_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(self.args.gnn_dp_rate, inplace=True) for i in range(self.hop_layers)])
        
    def forward(self, sent_embedding, enc_sent_padding_mask):
        
        mask = enc_sent_padding_mask.unsqueeze(-1) #.repeat(1, 1, sent_embedding.size(-1))
        sent_embedding = sent_embedding * mask
        
        d, d_0 = self.matrix_tree_layer(sent_embedding, enc_sent_padding_mask)
        
        d = (d.transpose(-1,-2) * mask).transpose(-1,-2) * mask
        d_0 = d_0 * enc_sent_padding_mask

        gnn_out = []
        for i in range(self.hop_layers):
            sent_embedding = self.msg_psg_layer[i](sent_embedding, d, d_0)
            sent_embedding = self.dropouts[i](sent_embedding)
            gnn_out.append(sent_embedding)
            
        return gnn_out, d, d_0

class HierGNNLayer(nn.Module):
    def __init__(self, args):
        super(HierGNNLayer, self).__init__()
        self.args = args
        self.matrix_tree_layer = Matrix_Tree_Layer(self.args)
        self.msg_psg_layer = Message_Passing_Layer(self.args)
        self.dropout = nn.Dropout(self.args.gnn_dp_rate, inplace=True)
        
    def forward(self, sent_embedding, enc_sent_padding_mask):
        
        mask = enc_sent_padding_mask.unsqueeze(-1) #.repeat(1, 1, sent_embedding.size(-1))
        sent_embedding = sent_embedding * mask
        
        d, d_0 = self.matrix_tree_layer(sent_embedding, enc_sent_padding_mask)
        
        d = (d.transpose(-1,-2) * mask).transpose(-1,-2) * mask
        d_0 = d_0 * enc_sent_padding_mask

        out = self.msg_psg_layer(sent_embedding, d, d_0)
        out = self.dropout(out)
        
        return out, d, d_0

class Message_Passing_Layer(nn.Module):
    def __init__(self, args):
        super(Message_Passing_Layer, self).__init__()
        self.args = args
        self.h_dim = 1024
        self.weight_gate = nn.Linear(2 * self.h_dim, self.h_dim)
        self.W_i = nn.Linear(self.h_dim, self.h_dim)
        self.layer_norm = nn.LayerNorm(self.h_dim)
        
    def forward(self, h, int_mat, root_ps):
        """
        Input: 
        h: hidden representation for each node (sentence embedding): (bs, MAX_NUM_SENT, sent_emb_dim)
        int_mat: interaction matrix: (bs, MAX_NUM_SENT, MAX_NUM_SENT)
        root_ps: roots probability: (bs, MAX_NUM_SENT)
        
        Eqt:
        𝑠_𝑖^((𝑙+1))  =(1−𝑎_𝑖^𝑟)𝑠_𝑖^((𝑙))+(𝑎_𝑖^𝑟)∑_(𝑘=1)^𝐾▒〖𝑎_ki 𝑠_𝑘^((𝑙)) 〗
        """
        
        # Calculate the msg, guided by interaction matrix (Adj)
        msg = torch.matmul(int_mat, h) # (bs, sent_num, sent_emb_dim)
        
        root_ps = root_ps.unsqueeze(-1)
        
        # structured guidance - large root prob, receive more msg
        update = root_ps * msg + (1.0-root_ps) * (self.W_i(h))
        
        # Gate mechanism by considering update and original representation
        gate = self.weight_gate(torch.cat((update, h), -1))
        gate = torch.sigmoid(gate)
        
        # 2. Update by gating
        h = gate * torch.tanh(update) + (1 - gate) * h
        h = self.layer_norm(h)
        
        return h
    
    
class Matrix_Tree_Layer(nn.Module):
    def __init__(self, args):
        super(Matrix_Tree_Layer, self).__init__()
        self.args = args
        self.str_dim_size = 1024
        self.h_dim = 1024
        
        # Projection for parent and child representation
        self.tp_linear = nn.Linear(self.h_dim, self.str_dim_size, bias=True)
        self.tc_linear = nn.Linear(self.h_dim, self.str_dim_size, bias=True)
        self.bilinear = BilinearMatrixAttention(self.str_dim_size, self.str_dim_size, use_input_biases=False, label_dim=1)
        
        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)
        
    def forward(self, sent_vecs, enc_sent_padding_mask):
        
        batch_size = sent_vecs.shape[0]
        sent_num = sent_vecs.shape[1]
        
        tp = torch.relu(self.tp_linear(sent_vecs))
        tc = torch.relu(self.tc_linear(sent_vecs))
        
        # Using the bilinear attention to compute f_jk: fjk = u^T_k W_a u_j
        scores = self.bilinear(tp, tc).view(batch_size, sent_num, sent_num)
        root = self.fi_linear(tp).view(batch_size, sent_num)  # bs, SENT_NUM, 1
        
        # masking out diagonal elements, see Eqt 2.1
        mask = scores.new_ones((scores.size(1), scores.size(1))) - scores.new_tensor(torch.eye(scores.size(1), scores.size(1))).cuda()
        mask = mask.unsqueeze(0).expand(scores.size(0), mask.size(0), mask.size(1))
        
        if self.args.not_sparse:
            A_ij = torch.exp(torch.tanh(scores))
            A_ij = (A_ij.transpose(-1,-2) * enc_sent_padding_mask.unsqueeze(-1)).transpose(-1,-2) * enc_sent_padding_mask.unsqueeze(-1) + 1e-6
            A_ij = A_ij * mask
            f_i = (torch.exp(torch.tanh(root)) * enc_sent_padding_mask) + 1e-6
        else:
            A_ij = torch.relu(scores)
            A_ij = (A_ij.transpose(-1,-2) * enc_sent_padding_mask.unsqueeze(-1)).transpose(-1,-2) * enc_sent_padding_mask.unsqueeze(-1) + 1e-6
            A_ij = A_ij * mask
            f_i = (torch.relu(root) * enc_sent_padding_mask) + 1e-6
    
        tmp = torch.sum(A_ij, dim=1)
        res = A_ij.new_zeros((batch_size, sent_num, sent_num)).cuda() #.to(self.device)
        #tmp = torch.stack([torch.diag(t) for t in tmp])
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)

        L_ij = -A_ij + res   #A_ij has 0s as diagonals

        L_ij_bar = L_ij.clone()
        L_ij_bar[:,0,:] = f_i
        
        LLinv = None
        LLinv = torch.inverse(L_ij_bar)

        d0 = f_i * LLinv[:,:,0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1,2) * LLinv_diag ).transpose(1,2)
        tmp2 = A_ij * LLinv.transpose(1,2)

        temp11 = A_ij.new_zeros((batch_size, sent_num, 1))
        temp21 = A_ij.new_zeros((batch_size, 1, sent_num))

        temp12 = A_ij.new_ones((batch_size, sent_num, sent_num-1))
        temp22 = A_ij.new_ones((batch_size, sent_num-1, sent_num))

        mask1 = torch.cat([temp11,temp12],2).cuda() #.to(self.device)
        mask2 = torch.cat([temp21,temp22],1).cuda() #.to(self.device)

        # Eqt: P(zjk = 1) = (1 − δ(j, k))AjkL¯−1kk − (1 − δ(j, 1))AjkL¯−1
        dx = mask1 * tmp1 - mask2 * tmp2
        
        return dx, d0 

    
class BilinearMatrixAttention(nn.Module):
    """
    Computes attention between two matrices using a bilinear attention function.  This function has
    a matrix of weights ``W`` and a bias ``b``, and the similarity between the two matrices ``X``
    and ``Y`` is computed as ``X W Y^T + b``.
    Parameters
    ----------
    matrix_1_dim : ``int``
        The dimension of the matrix ``X``, described above.  This is ``X.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_2_dim : ``int``
        The dimension of the matrix ``Y``, described above.  This is ``Y.size()[-1]`` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : ``Activation``, optional (default=linear (i.e. no activation))
        An activation function applied after the ``X W Y^T + b`` calculation.  Default is no
        activation.
    use_input_biases : ``bool``, optional (default = False)
        If True, we add biases to the inputs such that the final computation
        is equivalent to the original bilinear matrix multiplication plus a
        projection of both inputs.
    label_dim : ``int``, optional (default = 1)
        The number of output classes. Typically in an attention setting this will be one,
        but this parameter allows this class to function as an equivalent to ``torch.nn.Bilinear``
        for matrices, rather than vectors.
    """
    def __init__(self,
                 matrix_1_dim: int,
                 matrix_2_dim: int,
                 use_input_biases: bool = False,
                 label_dim: int = 1) -> None:
        super(BilinearMatrixAttention, self).__init__()
        if use_input_biases:
            matrix_1_dim += 1
            matrix_2_dim += 1

        if label_dim == 1:
            self._weight_matrix = Parameter(torch.Tensor(matrix_1_dim, matrix_2_dim))
        else:
            self._weight_matrix = Parameter(torch.Tensor(label_dim, matrix_1_dim, matrix_2_dim))

        self._bias = Parameter(torch.Tensor(1))
        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:

        if self._use_input_biases:
            bias1 = matrix_1.new_ones(matrix_1.size()[:-1] + (1,))
            bias2 = matrix_2.new_ones(matrix_2.size()[:-1] + (1,))

            matrix_1 = torch.cat([matrix_1, bias1], -1)
            matrix_2 = torch.cat([matrix_2, bias2], -1)

        weight = self._weight_matrix
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
        final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
        return final.squeeze(1)