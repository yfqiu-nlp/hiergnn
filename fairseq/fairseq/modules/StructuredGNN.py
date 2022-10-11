from __future__ import unicode_literals, print_function, division

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys

# from modules.BilinearMatrixAttention import BilinearMatrixAttention
from .GNNLayer import HierGNNLayer, LsrHierGNNLayer


from numpy import random
import numpy as np
import itertools
import math 

class StrGNNLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.h_dim = 1024
        self.hop_layers = self.args.hop_layer
        if not self.args.lsr_reasoning:
            self.hier_gnn = nn.ModuleList([HierGNNLayer(self.args) for i in range(self.hop_layers)])
        else:
            self.hier_gnn = LsrHierGNNLayer(self.args)
        self.gnn_fuse_layer = nn.Linear(self.h_dim * self.hop_layers, self.h_dim) # aggregate each layer output
        self.LayerName = 'StructuredGNN'
        
    def forward(self, x, encoder_padding_mask, segments):
        
        x = x.float()
        encoder_padding_mask = encoder_padding_mask.float()
        
        sent_vecs, sent_padding_mask, sent_lens = self.obtain_sent_vecs(x, encoder_padding_mask, segments) # sent pad mask: keep 1 as true
        # x = x.transpose(0,1) # T, B, C -> B, T, C
        res = sent_vecs.clone()
        
        # N-hop reasoning
        if not self.args.lsr_reasoning:
            # lir
            gnn_out, gnn_int_m, gnn_r_prob = [], [], []
            for i in range(self.hop_layers):
                sent_vecs, int_m, r_prob = self.hier_gnn[i](sent_vecs, sent_padding_mask)
                gnn_out.append(sent_vecs)
                gnn_int_m.append(int_m.unsqueeze(1))
                gnn_r_prob.append(r_prob.unsqueeze(1))

            gnn_out = torch.cat(gnn_out, dim=-1)
            gnn_int_m = torch.cat(gnn_int_m, dim=1) 
            gnn_r_prob = torch.cat(gnn_r_prob, dim=1)
        else:
            # Lsr
            gnn_out, gnn_int_m, gnn_r_prob = self.hier_gnn(sent_vecs, sent_padding_mask)
            gnn_out = torch.cat(gnn_out, dim=-1)
        
        gnn_out = F.relu_(self.gnn_fuse_layer(gnn_out)) + res # # bs, num_sent, 2 * hidden_dim
        gnn_out = gnn_out * sent_padding_mask.unsqueeze(-1)
    
        gnn_out = gnn_out.transpose(0,1) # bs, 100, dim
        gnn_out = gnn_out.half()
        
        return gnn_out, gnn_int_m, gnn_r_prob, ~sent_padding_mask, sent_lens # mark pad need to be mask as true
    
    def obtain_sent_vecs(self, x, encoder_padding_mask, segments):
        
        # Obtain sentence modeling
        ### x             [len, b, h]
        ### sent_x        [len, b, h]
        ### attn_weights  [head, b, len, len]
        ### segments      [b, len]
        
        sentence_x = []
        sentences_len = []
        for b_idx in range(segments.shape[0]): # loop each sample
            segments_idx = torch.nonzero(segments[b_idx])
            sent_len = segments_idx.shape[0]
            tmp_sentence_x = []
            for w_idx in range(sent_len):
                front = segments_idx[w_idx]
                if w_idx == sent_len - 1:
                    back = -1
                else:
                    back = segments_idx[w_idx+1]
                
                sent_inc = x[front:back, b_idx, :]
                
                #max_sent_vec = torch.max(sent_inc, 0, keepdim=True)   # [valuse, indices]    # max pooling
                #tmp_sentence_x.append(max_sent_vec[0])
                max_sent_vec = torch.mean(sent_inc, 0, keepdim=True)   # [valuse, indices]  # avg pooling
                tmp_sentence_x.append(max_sent_vec)
                
            if len(tmp_sentence_x) == 0:
                continue
            elif len(tmp_sentence_x) == 1:
                tmp_sentence_x = tmp_sentence_x[0]
            else:
                tmp_sentence_x = torch.cat(tmp_sentence_x)
                
            if not len(tmp_sentence_x) == 0:
                sentence_x.append(tmp_sentence_x)
                sentences_len.append(sentence_x[b_idx].shape[0])

        if len(tmp_sentence_x) == 0:
            sentences = x
            #sentences = sent_x
            sent_padding_mask = encoder_padding_mask
        else:
            sentences = torch.nn.utils.rnn.pad_sequence(sentence_x, batch_first=True)    # [B, s_len, hidden]
            sent_padding_mask = torch.ge(-torch.abs(sentences[:, :, 0]), 0.0)
            # sentences = sentences.transpose(0, 1)                                        # [s_len, B, hidden]
        
        return sentences, ~sent_padding_mask, sentences_len