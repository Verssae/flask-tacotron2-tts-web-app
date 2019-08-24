import os
import numpy as np
import sys
import time
import argparse
import torch
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from playsound import playsound
from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence
sys.path.append('waveglow/')
from waveglow.mel2samp import MAX_WAV_VALUE
from denoiser import Denoiser
import json

def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model

def plot_data(data, outdir, filename, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom')
    plot_path = os.path.join(outdir, 'plots', filename + '.png')
    plt.savefig(plot_path)
    print("plot saved at: {}".format(plot_path))

def tts(text, outdir, filename, lang, play=False, plot=True):
    if lang == 'en':
        cleaner = 'english_cleaners'
    else:
        cleaner = 'transliteration_cleaners'
    sequence = np.array(text_to_sequence(text, [cleaner]))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel, _, alignments = model.inference(sequence)
    # plot_spectrogram_to_numpy(mel_outputs)
    # plot_alignment_to_numpy(alignments)
    plot_data((mel_outputs.float().data.cpu().numpy()[0], mel.float().data.cpu().numpy()[0], alignments.float().data.cpu().numpy()[0].T),outdir, filename)
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=0.666)
        audio = audio * MAX_WAV_VALUE
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    audio_path = os.path.join(
        outdir, 'wavs', "{}.wav".format(filename))
    write(audio_path, hparams.sampling_rate, audio)
    if play:
        playsound(audio_path)
    print("audio saved at: {}".format(audio_path))

def ready(lang):
    hparams = create_hparams()
    hparams.sampling_rate = 22050

if __name__ == '__main__':

    hparams = create_hparams()
    hparams.sampling_rate = 22050

    parser = argparse.ArgumentParser(description='tts inference test')

    parser.add_argument('--lang', type=str,  default='kr',
                            choices=['kr', 'en'], help='모델 학습 언어 (kr/en)')
    parser.add_argument('--filename', '-n', type=str,  default=None, help='wav 파일 이름 (기본값:시간, 반복 시에는 기본값으로)')
    parser.add_argument('model',  type=str,  help='학습된 모델 경로')
    parser.add_argument('text', type=str, nargs='+', help='text')
    parser.add_argument('--repeat', '-r', type=bool, default=False, help='반복모드? 종료는 엔터')
    parser.add_argument('--plot', '-p', type=bool, default=True, help='mel output plot?')
    parser.add_argument('--play', '-a', type=bool, default=False, help='mel output plot?')

    opts = parser.parse_args()

    text = ' '.join(opts.text)
    checkpoint_path = opts.model
    lang = opts.lang
    outdir = 'tts'
    rep = opts.repeat
    filename = opts.filename if opts.filename else str(time.time())
    plot = opts.plot
    play = opts.play

    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()

    waveglow_path = 'waveglow_256channels.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()

    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
            
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    tts(text, outdir, filename, lang, play, plot)
    while rep:
        text = input('>')
        if len(text) < 1:
            break
        tts(text, outdir, str(time.time()), lang, play, plot)
    print("exit.")
