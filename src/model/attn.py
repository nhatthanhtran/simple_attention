import torch
import torch.nn as nn


from math import sqrt


Class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,attention_dropout=0.1, output_attention = False):
        super(FullAttention,self).__init__()
        self.scale = scale
        self.mask_flag = mast_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self,queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = keys.shape
        # from attention paper sqrt of dimension of the keys
        scale = self.scale or 1./sqrt(D)

        # compute the QK'
        score = torch.einsum("blhe,bshe->bhls",queries, keys)
        
        #compute softmax(QK'/sqrt(d)), then drop out as well
        A = self.dropout(torch.softmax(score*scale, dim=-1))
        
        #compute softmax(QK'/sqrt(d))V
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(),A)
        else:
            return (V.contiguous, None)


    
