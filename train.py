from config import Config
from TranslationModel import TranslationModel
from data_helpers import LoadEnglishGermanDataset, my_tokenizer
import torch
import time
import torch.nn as nn
import os


class CustomSchedule(nn.Module):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.step = 1.

    def __call__(self):
        arg1 = self.step ** -0.5
        arg2 = self.step * (self.warmup_steps ** -1.5)
        self.step += 1.
        return (self.d_model ** -0.5) * min(arg1, arg2)


def accuracy(logits, y_true, PAD_IDX):
    """
    :param logits:  [tgt_len,batch_size,tgt_vocab_size]
    :param y_true:  [tgt_len,batch_size]
    :param PAD_IDX:
    :return:
    """
    y_pred = logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [tgt_len,batch_size,tgt_vocab_size] 转成 [batch_size, tgt_len,tgt_vocab_size]
    y_true = y_true.transpose(0, 1).reshape(-1)
    # 将 [tgt_len,batch_size] 转成 [batch_size， tgt_len]
    acc = y_pred.eq(y_true)  # 计算预测值与正确值比较的情况
    mask = torch.logical_not(y_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    acc = acc.logical_and(mask)  # 去掉acc中mask的部分
    correct = acc.sum().item()
    total = mask.sum().item()
    return float(correct) / total, correct, total


def train_model(config):
    data_loader = LoadEnglishGermanDataset(config.train_corpus_file_paths,
                                           batch_size=config.batch_size,
                                           tokenizer=my_tokenizer,
                                           min_freq=config.min_freq)
    train_iter, valid_iter, test_iter = \
        data_loader.load_train_val_test_data(config.train_corpus_file_paths,
                                             config.val_corpus_file_paths,
                                             config.test_corpus_file_paths)
    translation_model = TranslationModel(src_vocab_size=len(data_loader.de_vocab),
                                         tgt_vocab_size=len(data_loader.en_vocab),
                                         d_model=config.d_model,
                                         nhead=config.num_head,
                                         num_encoder_layers=config.num_encoder_layers,
                                         num_decoder_layers=config.num_decoder_layers,
                                         dim_feedforward=config.dim_feedforward,
                                         dropout=config.dropout)
    for p in translation_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model_save_path = os.path.join(config.model_save_dir, 'model.pkl')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        translation_model.load_state_dict(loaded_paras)
        print("## 成功载入已有模型，进行追加训练......")
    translation_model = translation_model.to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_loader.PAD_IDX)
    learning_rate = CustomSchedule(config.d_model)
    optimizer = torch.optim.Adam(translation_model.parameters(),
                                 lr=0.,
                                 betas=(config.beta1, config.beta2), eps=config.epsilon)
    translation_model.train()
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (src, tgt) in enumerate(train_iter):
            src = src.to(config.device)  # [src_len, batch_size]
            tgt = tgt.to(config.device)
            tgt_input = tgt[:-1, :]  # 解码部分的输入, [tgt_len,batch_size]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask \
                = data_loader.create_mask(src, tgt_input, config.device)
            logits = translation_model(
                src=src,  # Encoder的token序列输入，[src_len,batch_size]
                tgt=tgt_input,  # Decoder的token序列输入,[tgt_len,batch_size]
                src_mask=src_mask,  # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的
                tgt_mask=tgt_mask,
                # Decoder的注意力Mask输入，用于掩盖当前position之后的position [tgt_len,tgt_len]
                src_key_padding_mask=src_padding_mask,  # 用于mask掉Encoder的Token序列中的padding部分
                tgt_key_padding_mask=tgt_padding_mask,  # 用于mask掉Decoder的Token序列中的padding部分
                memory_key_padding_mask=src_padding_mask)  # 用于mask掉Encoder的Token序列中的padding部分
            # logits 输出shape为[tgt_len,batch_size,tgt_vocab_size]
            optimizer.zero_grad()
            tgt_out = tgt[1:, :]  # 解码部分的真实值  shape: [tgt_len,batch_size]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            # [tgt_len*batch_size, tgt_vocab_size] with [tgt_len*batch_size, ]
            loss.backward()
            lr = learning_rate()
            for p in optimizer.param_groups:
                p['lr'] = lr
            optimizer.step()
            losses += loss.item()
            acc, _, _ = accuracy(logits, tgt_out, data_loader.PAD_IDX)
            print(f"Epoch: {epoch}, Train loss :{loss.item():.3f}, Train acc: {acc}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if epoch % 2 == 0:
            acc = evaluate(config, valid_iter, translation_model, data_loader)
            print(f"Accuracy on validation{acc:.3f}")
            torch.save(translation_model.state_dict(), model_save_path)


def evaluate(config, valid_iter, model, data_loader):
    model.eval()
    correct, totals = 0, 0
    for idx, (src, tgt) in enumerate(valid_iter):
        src = src.to(config.device)
        tgt = tgt.to(config.device)
        tgt_input = tgt[:-1, :]  # 解码部分的输入

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
            data_loader.create_mask(src, tgt_input, device=config.device)

        logits = model(src=src,  # Encoder的token序列输入，
                       tgt=tgt_input,  # Decoder的token序列输入
                       src_mask=src_mask,  # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的
                       tgt_mask=tgt_mask,  # Decoder的注意力Mask输入，用于掩盖当前position之后的position
                       src_key_padding_mask=src_padding_mask,  # 用于mask掉Encoder的Token序列中的padding部分
                       tgt_key_padding_mask=tgt_padding_mask,  # 用于mask掉Decoder的Token序列中的padding部分
                       memory_key_padding_mask=src_padding_mask)  # 用于mask掉Encoder的Token序列中的padding部分
        tgt_out = tgt[1:, :]  # 解码部分的真实值  shape: [tgt_len,batch_size]
        _, c, t = accuracy(logits, tgt_out, data_loader.PAD_IDX)
        correct += c
        totals += t
    model.train()
    return float(correct) / totals


if __name__ == '__main__':
    config = Config()
    train_model(config)
