# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import torchaudio
import random
import torch
import os
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
import librosa
import torch
import torchaudio
from scipy.signal import hilbert
from pathlib import Path
from audiocraft.metrics import CLAPTextConsistencyMetric
from diffusers import AudioLDMPipeline




class MusicGenSmall:
    def __init__(self, model_name="facebook/musicgen-small"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    def generate_music(self, prompt):
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        audio_values = self.model.generate(**inputs, max_new_tokens=1503)
        return audio_values[0, 0].numpy()
    
class MusicGenMedium:
    def __init__(self, model_name="facebook/musicgen-medium"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    def generate_music(self, prompt):
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        audio_values = self.model.generate(**inputs, max_new_tokens=1503)
        print(audio_values[0, 0])
        return audio_values[0, 0].numpy()

class MusicGenLarge:
    def __init__(self, model_name="facebook/musicgen-large"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    def generate_music(self, prompt):
        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        audio_values = self.model.generate(**inputs, max_new_tokens=1503)
        return audio_values[0, 0].numpy()
