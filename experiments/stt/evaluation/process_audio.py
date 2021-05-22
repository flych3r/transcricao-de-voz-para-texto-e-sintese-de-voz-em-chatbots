from pydub import AudioSegment
import os
import glob
from rich.progress import track

def preprocess_audio(audio):
    return audio.set_sample_width(2).set_channels(1).set_frame_rate(48000)


def read_audio_file_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = preprocess_audio(audio)
    file_path = os.path.splitext(file_path)[0]
    audio.export(f'{file_path}.wav', format='wav')
    return audio

mozilla = glob.glob('./mozilla_processed/wav/*.mp3')
voxforge = glob.glob('./voxforge_processed/wav/*.wav')


for fp in track(mozilla, description=f'Converting mozilla'):
    read_audio_file_to_wav(fp)

for fp in track(voxforge, description=f'Converting voxforge'):
    read_audio_file_to_wav(fp)
