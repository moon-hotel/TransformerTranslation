import torch
import torch.nn as nn

if __name__ == '__main__':
    h, d_model, seq_len = 6, 768, 7
    head_dim, bsz = d_model // h, 2
    x = torch.randn([seq_len, bsz, d_model])
    q_pro = nn.Linear(d_model, d_model)
    k_pro = nn.Linear(d_model, d_model)
    v_pro = nn.Linear(d_model, d_model)
    out_pro = nn.Linear(d_model, d_model)
    q, k, v = q_pro(x), k_pro(x), v_pro(x)
    # [seq_len, bsz, head_dim*h]
    scaling = float(head_dim) ** -0.5
    q = q * scaling
    q = q.view(seq_len, bsz * h, head_dim).transpose(0, 1)
    k = k.view(seq_len, bsz * h, head_dim).transpose(0, 1)
    v = v.view(seq_len, bsz * h, head_dim).transpose(0, 1)
    # [ bsz*h, seq_len, head_dim] [12,7,128]
    weight = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
    # [bsz*h, seq_len, seq_len]  [12,7,7]
    z = (weight @ v)
    # [12,7,7] @ [12,7,128] = [12,7,128]
    # [bsz*h, seq_len, head_dim]
    z = z.transpose(0, 1).reshape(seq_len, bsz, d_model)
    # [seq_len, bsz, emb_dim] [7,2,768]
    out = out_pro(z)
    # [seq_len, bsz, emb_dim] [7,2,768]
    print(out.shape)
