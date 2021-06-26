from config import Config
from TranslationModel import TranslationModel
from data_helpers import LoadEnglishGermanDataset, my_tokenizer
import torch



def greedy_decode(model, src, max_len, start_symbol, config, data_loader):
    src = src.to(config.device)
    memory = model.encoder(src)  # 对输入的Token序列进行解码翻译
    ys = torch.ones(1, 1).fill_(start_symbol). \
        type(torch.long).to(config.device)  # 解码的第一个输入，起始符号
    for i in range(max_len - 1):
        memory = memory.to(config.device)
        tgt_mask = (model.my_transformer.generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(config.device)  # 根据tgt_len产生一个注意力mask矩阵（对称的）
        out = model.decoder(ys, memory, tgt_mask)  # [tgt_len,tgt_vocab_size]
        out = out.transpose(0, 1)  # [tgt_vocab_size, tgt_len]
        prob = model.classification(out[:, -1])  # 只对对预测的下一个词进行分类
        _, next_word = torch.max(prob, dim=1)  # 选择概率最大者
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # 将当前时刻解码的预测输出结果，同之前所有的结果堆叠作为输入再去预测下一个词。
        if next_word == data_loader.EOS_IDX:  # 如果当前时刻的预测输出为结束标志，则跳出循环结束预测。
            break
    return ys


def translate(model, src, data_loader, config):
    src_vocab = data_loader.de_vocab
    tgt_vocab = data_loader.en_vocab
    src_tokenizer = data_loader.tokenizer
    model.eval()
    tokens = [src_vocab.stoi[tok] for tok in src_tokenizer(src)]  # 构造一个样本
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))  # 将src_len 作为第一个维度
    tgt_tokens = greedy_decode(model, src, max_len=num_tokens + 5,
                               start_symbol=data_loader.BOS_IDX, config=config,
                               data_loader=data_loader).flatten()  # 解码的预测结果
    return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")


def translate_german_to_english(src, config):
    data_loader = LoadEnglishGermanDataset(config.train_corpus_file_paths,
                                           batch_size=config.batch_size,
                                           tokenizer=my_tokenizer)
    translation_model = TranslationModel(src_vocab_size=len(data_loader.de_vocab),
                                         tgt_vocab_size=len(data_loader.en_vocab),
                                         d_model=config.d_model,
                                         nhead=config.num_head,
                                         num_encoder_layers=config.num_encoder_layers,
                                         num_decoder_layers=config.num_decoder_layers,
                                         dim_feedforward=config.dim_feedforward,
                                         dropout=config.dropout)
    translation_model = translation_model.to(config.device)
    torch.load(config.model_save_dir + '/model.pkl')

    r = translate(translation_model, src, data_loader, config)
    return r

if __name__ == '__main__':
    srcs = ["Eine Gruppe von Menschen steht vor einem Iglu.",
            "Ein Mann in einem blauen Hemd steht auf einer Leiter und putzt ein Fenster."]
    tgts = ["A group of people are facing an igloo.",
            "A man in a blue shirt is standing on a ladder cleaning a window."]
    config = Config()
    for i, src in enumerate(srcs):
        r = translate_german_to_english(src, config)
        print(f"德语：{src}")
        print(f"翻译：{r}")
        print(f"英语：{tgts[i]}")
