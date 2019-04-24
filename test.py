import torch
import numpy as np
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from network import *
from utils import audio
from text import text_to_sequence
import hparams
from gen_training_loader import get_separator
from bert_embedding import get_bert_embedding
from train import get_embedding_input


def get_BERT_embeddings(model_BERT, model_Tokenizer, text, seq_size):
    embeddings, tokens = get_bert_embedding(text, model_BERT, model_Tokenizer)
    sep_list = get_separator(text, tokens)
    em_info = (embeddings, sep_list)
    em_info = [em_info]

    embeddings = get_embedding_input(em_info, seq_size)
    return embeddings


def synthesizer(model_Tacotron, model_BERT, model_Tokenizer, text, device):
    seq = text_to_sequence(text, [hparams.cleaners])
    # seq = torch.Tensor(seq).to(device)
    seq = np.stack([seq])
    if torch.cuda.is_available():
        seq = torch.from_numpy(seq).type(torch.cuda.LongTensor).to(device)
    else:
        seq = torch.from_numpy(seq).type(torch.LongTensor).to(device)
    # print(seq)

    embeddings = get_BERT_embeddings(
        model_BERT, model_Tokenizer, text, seq.size(1))
    embeddings = embeddings.to(device)

    # Provide [GO] Frame
    mel_input = np.zeros(
        [np.shape(seq)[0], hparams.num_mels, 1], dtype=np.float32)
    mel_input = torch.Tensor(mel_input).to(device)
    # print(np.shape(mel_input))

    model_Tacotron.eval()
    with torch.no_grad():
        _, linear_output = model_Tacotron(seq, mel_input, embeddings)
        # print(np.shape(linear_output))

    # trans_linear = audio.trans(linear_output[0].cpu().numpy())
    wav = audio.inv_spectrogram(linear_output[0].cpu().numpy())
    # print(audio.find_endpoint(wav))
    # print(np.shape(wav))
    wav = wav[:audio.find_endpoint(wav)]
    # print(np.shape(wav))
    return wav


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = nn.DataParallel(Tacotron()).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    print("Models Have Been Defined")

    # Load checkpoint
    checkpoint = torch.load(os.path.join(
        hparams.checkpoint_path, 'checkpoint_100000.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    print("Sucessfully Loaded.")

    text = "How are you?"
    wav = synthesizer(model, model_bert, tokenizer, text, device)
    audio.save_wav(wav, "test.wav")
