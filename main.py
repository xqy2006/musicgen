import sys
import os
input = sys.argv[1].split("\n")[-1]
print(input,sys.argv[2])
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained('small')
model.set_generation_params(duration=10)  # generate 8 seconds.
descriptions = [input]
wav = model.generate(descriptions)  # generates 3 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(sys.argv[2], one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
os.system("mkdir .\\result\\music")
os.system("ls")
os.system("cp "+sys.argv[2]+".wav ./result/music/")

