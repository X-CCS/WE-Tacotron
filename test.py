import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import network

def test(model_tacotron, model_bert, model_tokenizer, text):
    tokenized = model_tokenizer(text)