import io
import os
import sys
import textwrap
import time

import numpy as np
import pydub
import torch
from pydub import AudioSegment
from TTS.models.tacotron import Tacotron
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config
from TTS.utils.text import phoneme_to_sequence, text_to_sequence
from TTS.utils.text.symbols import phonemes, symbols
from WaveRNN.models.wavernn import Model

TTS_PATH = './TTS'
WAVERNN_PATH = './WaveRNN'
# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally
sys.path.append(WAVERNN_PATH) # set this if WaveRNN is not installed globally

sys.path.append('')



def synthesis(m, s, CONFIG, use_cuda, ap, language=None):
    """ Given the text, synthesising the audio """
    if language is None:
      language=CONFIG.phoneme_language
    text_cleaner = [CONFIG.text_cleaner]
    # print(phoneme_to_sequence(s, text_cleaner))
    # print(sequence_to_phoneme(phoneme_to_sequence(s, text_cleaner)))
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(s, text_cleaner, language),
            dtype=np.int32
        )
    else:
        seq = np.asarray(text_to_sequence(s, text_cleaner), dtype=np.int32)
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    if use_cuda:
        chars_var = chars_var.cuda()
    mel_spec, linear_spec, alignments, stop_tokens = m.forward(
        chars_var.long()
    )
    linear_spec = linear_spec[0].data.cpu().numpy()
    mel_spec = mel_spec[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    mel_tensor = torch.FloatTensor(mel_spec.T).unsqueeze(0)
    if torch.cuda.is_available():
        mel_tensor = mel_tensor.cuda()
    wav = wavernn.generate(mel_tensor, batched=True, target=11000, overlap=550)
    return wav


def tts(model, text, CONFIG, use_cuda, ap, language=None):
    waveform = synthesis(model, text, CONFIG, use_cuda, ap, language=language)
    return waveform


def process_audio(audio):
    audio = sum(pydub.silence.split_on_silence(audio, silence_thresh=-36))
    return audio.set_sample_width(2).set_channels(1).set_frame_rate(22500)


def synthesise_text(sentence):
    start = time.time()
    wavs = []
    for frase in textwrap.wrap(sentence, width=br, break_long_words=False):
        f = io.BytesIO()
        wav = tts(model, ' ' + frase + ' ', CONFIG, use_cuda, ap2)
        ap2.save_wav(wav, f)
        wavs.append(f)
    wavs = [AudioSegment.from_file_using_temporary_files(w) for w in wavs]
    wav = sum(wavs)
    wav = process_audio(wav)
    fp = 'audio'
    wav.export('{}.wav'.format(fp), format='wav')
    end = time.time()
    print('\n', end - start, 'segundos')


MODEL_PATH = 'checkpoint.pth.tar'
CONFIG_PATH =  'TTS/config.json'
OUT_FOLDER = 'samples/'
try:
    os.mkdir(OUT_FOLDER)
except:
    pass

CONFIG = load_config(CONFIG_PATH)
use_cuda = torch.cuda.is_available()

VOCODER_MODEL_PATH = 'WaveRNN/saver.pth.tar'
VOCODER_CONFIG_PATH = 'WaveRNN/config_16K.json'
VOCODER_CONFIG = load_config(VOCODER_CONFIG_PATH)

# load the model
ap2 = AudioProcessor(**VOCODER_CONFIG.audio)
ap = AudioProcessor(**CONFIG.audio)

num_chars = len(phonemes) if CONFIG.use_phonemes else len(symbols)
model= Tacotron(num_chars, CONFIG.embedding_size, ap.num_freq, ap.num_mels, CONFIG.r, CONFIG.memory_size)

# load model state
if use_cuda:
    cp = torch.load(MODEL_PATH)
else:
    cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()

bits = 10

wavernn = Model(
    rnn_dims=512,
    fc_dims=512,
    mode=VOCODER_CONFIG.mode,
    mulaw=VOCODER_CONFIG.mulaw,
    pad=VOCODER_CONFIG.pad,
    use_aux_net=VOCODER_CONFIG.use_aux_net,
    use_upsample_net=VOCODER_CONFIG.use_upsample_net,
    upsample_factors=VOCODER_CONFIG.upsample_factors,
    feat_dims=80,
    compute_dims=128,
    res_out_dims=128,
    res_blocks=10,
    hop_length=ap2.hop_length,
    sample_rate=ap2.sample_rate,
)

check = torch.load(VOCODER_MODEL_PATH, map_location=torch.device('cuda' if use_cuda else 'cpu'))
wavernn.load_state_dict(check['model'])
if use_cuda:
    wavernn.cuda()
wavernn.eval()


model.decoder.max_decoder_steps = 500
br = 50
stay = True
while(stay):
    synthesise_text(input('Type a sentence to be synthesised > '))
    stay = not (input('Type exit to stop > ') == 'exit')
