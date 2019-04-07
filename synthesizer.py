from network import *
from utils import audio
from text import text_to_sequence
import hparams

import numpy as np
import os


def synthesizer(model, text, device):
    seq = text_to_sequence(text, [hparams.cleaners])
    # seq = torch.Tensor(seq).to(device)
    seq = np.stack([seq])
    if torch.cuda.is_available():
        seq = torch.from_numpy(seq).type(torch.cuda.LongTensor).to(device)
    else:
        seq = torch.from_numpy(seq).type(torch.LongTensor).to(device)
    # print(seq)

    # Provide [GO] Frame
    mel_input = np.zeros(
        [np.shape(seq)[0], hparams.num_mels, 1], dtype=np.float32)
    mel_input = torch.Tensor(mel_input).to(device)
    # print(np.shape(mel_input))

    model.eval()
    with torch.no_grad():
        _, linear_output = model(seq, mel_input)
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
    print("Model Have Been Defined")

    # Load checkpoint
    checkpoint = torch.load(os.path.join(
        hparams.checkpoint_path, 'checkpoint_40.pth.tar'))
    model.load_state_dict(checkpoint['model'])

    text = "in being comparatively modern."
    wav = synthesizer(model, text, device)
    audio.save_wav(wav, "test.wav")
