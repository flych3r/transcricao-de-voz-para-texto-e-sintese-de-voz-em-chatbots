import io
import os
import sys
import textwrap
import time

import numpy as np
import pydub
import tensorflow as tf
from pydub import AudioSegment
from scipy.io.wavfile import write
from TTS_Conv.data_load import load_vocab, text_normalize
from TTS_Conv.hyperparams import Hyperparams as hp
from TTS_Conv.train import Graph
from TTS_Conv.utils import *

TTS_PATH = './TTS_Conv'
# add libraries into environment
sys.path.append(TTS_PATH) # set this if TTS is not installed globally



# Load graph
g = Graph(mode='synthesize'); print('Graph loaded')

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# Restore parameters
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
saver1 = tf.train.Saver(var_list=var_list)
saver1.restore(sess, os.path.join('saver-text','text2mel','saver'))
print('Text2Mel Restored!')

var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
saver2 = tf.train.Saver(var_list=var_list)
saver2.restore(sess, os.path.join('saver-text','mel2linear','saver'))
print('SSRN Restored!')

def process_audio(audio):
    audio = sum(pydub.silence.split_on_silence(audio, silence_thresh=-36))
    return audio.set_sample_width(2).set_channels(1).set_frame_rate(22500)

def sysntesise_text(sentence):
    start = time.time()
    # print('input text: ',frase)
    wavs = []
    for frase in textwrap.wrap(sentence, width=100, break_long_words=False):
        frase = '1 '+frase # add factor
        #normalize remove inavalid characters
        frase = text_normalize(frase.split(' ', 1)[-1]).strip() + 'E' # text normalization, E: EOS

        # print('normalized text:',frase)

        char2idx, idx2char = load_vocab()

        #convert characters to numbers
        text = np.zeros((1, hp.max_N), np.int32) #hp.max_N = 128, is the max number for characters
        text[0, :len(frase)] = [char2idx[char] for char in frase]

        # print('converted text:',text)

        L = text
        # Feed Forward
        ## mel
        # note: hp.max_T can be changed depending on the phrase to be synthesized, the default value is 210, which generates an audio of maximum 10 seconds, if it decreases this value can obtain a greater speed of synthesis.
        hp.max_T = 210
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)
        for j in range(hp.max_T):
            _gs, _Y, _max_attentions, _alignments = sess.run([g.global_step, g.Y, g.max_attentions, g.alignments], {g.L: L,g.mels: Y, g.prev_max_attentions: prev_max_attentions})
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]

        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        for i, mag in enumerate(Z):
            # print("Working on file", i+1)
            wav = spectrogram2wav(mag)
            f = io.BytesIO()
            write(f, hp.sr, wav)
            #save for frase.wav
            wavs.append(f)

    wavs = [AudioSegment.from_file_using_temporary_files(w) for w in wavs]
    wav = sum(wavs)
    wav = process_audio(wav)
    wav.export('audio.wav', format='wav')
    end = time.time()
    print(end - start, 'segundos')

stay = True
while(stay):
    sysntesise_text(input('Type a sentence to be synthesised > '))
    stay = not (input('Type exit to stop > ') == 'exit')
