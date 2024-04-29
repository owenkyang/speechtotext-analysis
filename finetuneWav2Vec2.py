import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import pandas as pd
import numpy as np
import os
from datasets import load_dataset
import torch
from torch.optim import AdamW
import soundfile as sf

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

datafame = pd.read_excel()
def preprocess(audio_path):
    audio_input = sf.read()
    }

def load_audio(filename, sample_rate=16000):
    audio, rate = librosa.load(filename, sr = sample_rate)
    return audio

def fineTune(model, audio_paths, transcriptions, epochs = 10):
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e -5) # Adam optimizer
    loss_function = torch.nn.CrossEntropyLoss() #Cross Entropy cost function

    model.train()

    for epoch in range(epochs):
