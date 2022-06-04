# -*- coding: utf-8 -*- #

import jieba
from torchtext.vocab import Vocab, vocab
from collections import Counter, OrderedDict


import pandas as pd
import torch
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.datasets import imdb
from torch.nn import init
from tqdm import tqdm
import nltk


def tokenizer(s, word=False):
    """
    :param s:
    :param word:  True表示按字切分
    :return:
    """
    if word:
        r = [w for w in s]
    else:
        s = jieba.cut(s, cut_all=False)  # 普通分词结果
        r = " ".join(s).split()
    return r


def build_vocab(tokenizer, filepath, min_freq, specials=None):
    if specials is None:
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    counter = Counter()
    with open(filepath, encoding='utf8') as f:
        for string_ in f:
            counter.update(tokenizer(string_.strip(), word=True))

    print(f"counter:{counter}")  # 计数

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # 构造成可接受的格式：[(单词,num), ...]
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    return vocab(ordered_dict, min_freq=min_freq, specials=specials)


def main():
    s = "问君能有几多愁？恰似一江春水向东流。"
    result = tokenizer(s=s, word=True)
    print(result)

    filepath = '../data/tokenizer_test.txt'
    my_vocab = build_vocab(tokenizer, filepath, min_freq=1,)

    print(my_vocab['会'])

    print("word->token：", my_vocab.get_stoi())

    print("token->word：", my_vocab.get_itos())

    pass


if __name__ == '__main__':
    main()