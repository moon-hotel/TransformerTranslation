import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_

from .Attention import MyMultiheadAttention

is_print_shape = True


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 ):
        super(MyTransformer, self).__init__()

        """
        :param d_model:  d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:               多头注意力机制中多头的数量，论文默认为值 8
        :param num_encoder_layers:  encoder堆叠的数量，也就是论文中的N，论文默认值为6
        :param num_decoder_layers:  decoder堆叠的数量，也就是论文中的N，论文默认值为6
        :param dim_feedforward:     全连接中向量的维度，论文默认值为 2048
        :param dropout:             丢弃率，论文中的默认值为 0.1
        """

        #  ================ 编码部分 =====================
        encoder_layer = MyTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = MyTransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        # ================ 解码部分 =====================
        decoder_layer = MyTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MyTransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        :param src:   [src_len,batch_size,embed_dim]
        :param tgt:  [tgt_len, batch_size, embed_dim]
        :param src_mask:  None
        :param tgt_mask:  [tgt_len, tgt_len]
        :param memory_mask: None
        :param src_key_padding_mask: [batch_size, src_len]
        :param tgt_key_padding_mask: [batch_size, tgt_len]
        :param memory_key_padding_mask:  [batch_size, src_len]
        :return: [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        """
        memory = self.encoder(src, mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        return output

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # [sz,sz]


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1    
        
        
        """
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param src_mask:  None
        :param src_key_padding_mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              # 计算多头注意力
                              key_padding_mask=src_key_padding_mask, )[0]
        # src2: [src_len,batch_size,num_heads*kdim] num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # [src_len,batch_size,num_heads*kdim]

        # [src_len,batch_size,dim_feedforward]
        src2 = self.activation(self.linear1(src))
        # [src_len,batch_size,num_heads*kdim]
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        return src


class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        """
        encoder_layer: 就是包含有多头注意力机制的一个编码层
        num_layers: 克隆得到多个encoder layers 论文中默认为6
        norm: 归一化层
        
        """
        self.layers = _get_clones(
            encoder_layer, num_layers)  # 克隆得到多个encoder layers 论文中默认为6
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:# [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)  # 多个encoder layers层堆叠后的前向传播过程
        if self.norm is not None:
            output = self.norm(output)
        # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        return output


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1    
        """
        self.self_attn = MyMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head attention)
        self.multihead_attn = MyMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # 编码部分输出（memory）和解码部分之间的多头注意力机制。
        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt:  解码部分的输入，形状为 [tgt_len,batch_size, embed_dim]
        :param memory: 编码部分的输出（memory）, [src_len,batch_size,embed_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        tgt2 = self.self_attn(tgt, tgt, tgt,  # [tgt_len,batch_size, embed_dim]
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 解码部分输入序列之间'的多头注意力（也就是论文结构图中的Masked Multi-head attention)

        tgt = tgt + self.dropout1(tgt2)  # 接着是残差连接
        tgt = self.norm1(tgt)  # [tgt_len,batch_size, embed_dim]

        tgt2 = self.multihead_attn(tgt, memory, memory,  # [tgt_len, batch_size, embed_dim]
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        # 解码部分的输入经过多头注意力后同编码部分的输出（memory）通过多头注意力机制进行交互
        tgt = tgt + self.dropout2(tgt2)  # 残差连接
        tgt = self.norm2(tgt)  # [tgt_len, batch_size, embed_dim]

        # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.activation(self.linear1(tgt))
        # [tgt_len, batch_size, embed_dim]
        tgt2 = self.linear2(self.dropout(tgt2))
        # 最后的两层全连接
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        return tgt


class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt: 解码部分的输入，形状为 [tgt_len,batch_size, embed_dim]
        :param memory: 编码部分最后一层的输出 [src_len,batch_size, embed_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码，一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况，形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:
        """
        output = tgt  # [tgt_len,batch_size, embed_dim]

        for mod in self.layers:  # 这里的layers就是N层解码层堆叠起来的
            output = mod(output, memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        return output
