import sys

sys.path.append('../')
from model.TranslationModel import TranslationModel
import torch

if __name__ == '__main__':
    src_len = 7
    batch_size = 2
    dmodel = 32
    tgt_len = 8
    num_head = 4
    src = torch.tensor([[4, 3, 2, 6, 0, 0, 0],
                        [5, 7, 8, 2, 4, 0, 0]]).transpose(0, 1)  # 转换成 [src_len, batch_size]
    src_key_padding_mask = torch.tensor([[False, False, False, True, True, True, True],
                                         [False, False, True, True, True, True, True]])
    # Fasel表示当前位置不是padding，True表示当前位置是padding后的地方
    # True位置后续对应会被置为负无穷
    tgt = torch.tensor([[1, 3, 3, 5, 4, 3, 0, 0],
                        [1, 6, 8, 2, 9, 1, 0, 0]]).transpose(0, 1)
    tgt_key_padding_mask = torch.tensor([[False, False, False, False, False, False, True, True],
                                         [False, False, False, False, True, True, True, True]])

    trans_model = TranslationModel(src_vocab_size=10, tgt_vocab_size=15,
                                   d_model=dmodel, nhead=num_head, num_encoder_layers=6,
                                   num_decoder_layers=6, dim_feedforward=30, dropout=0.1)
    tgt_mask = trans_model.my_transformer.generate_square_subsequent_mask(tgt_len)
    logits = trans_model(src, tgt=tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=src_key_padding_mask)
    print(logits.shape)
