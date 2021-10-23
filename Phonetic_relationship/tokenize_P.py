
from re import S
import tokenizers
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from tokenizers import AddedToken
import collections
from pypinyin import pinyin,Style
import os

local_chinese_bert_path='/mnt/data10t/bakuphome20210617/ljh/model/bert-base-chinese/'

def is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

def get_pinyins(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    
    pinyins=[]

    for token in vocab.keys():
        for char in token:
            if is_chinese_char(ord(char)) and any(x in pinyin(char,style=Style.NORMAL)[0][0] for x in 'aoeiuvn'):
                pinyins.append(pinyin(char,style=Style.NORMAL)[0][0])
    pinyins=sorted({*pinyins})
    return pinyins



def get_phonetic_bert():
    config = BertConfig.from_pretrained(local_chinese_bert_path)
    tokenizer = BertTokenizer.from_pretrained(local_chinese_bert_path)
    model=BertForMaskedLM.from_pretrained(local_chinese_bert_path)
    num_added_tokens=tokenizer.add_tokens(['['+py+']' for py in get_pinyins(os.path.join(local_chinese_bert_path,'vocab.txt'))],special_tokens=True)
    print(f'{num_added_tokens} pinyins have been added to tokenizer')
    model.resize_token_embeddings(len(tokenizer))
    return config,tokenizer,model


if __name__ == "__main__":
    config,tokenizer,model=get_phonetic_bert()
    # print(tokenizer.unique_no_split_tokens)
    # print(tokenizer.added_tokens_encoder)
    print(model)
    text='这是 最常见 的 中文 bert 语言 模型，僧侣'
    tokens = tokenizer.tokenize(text)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.Tensor([ids]).long()
    print(ids)
    text='这是 最[chang]见的[MASK]文bert语言模[xing]'
    tokens = tokenizer.tokenize(text)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.Tensor([ids]).long()
    print(ids)
    # bert = BertModel.from_pretrained('../data/chinese_wwm_pytorch/pytorch_model.bin', config=config)
    # model = SoftMaskedBert(bert, tokenizer, 2, 1)
    # text = '中国的'
    
    # out = model(ids)
    # # out = bert(ids)
    # print(out)
