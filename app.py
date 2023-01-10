import streamlit as st
import keras
import os
import numpy as np
import random
import torch
import torchaudio
from torchaudio import transforms

st.set_option('deprecation.showfileUploaderEncoding', False)#Streamlit ka rr nahi hoga



class AudioProcessing():
    """
    open method is used to load the audio file and returns your signal as a Tensor and sampling rate
    Tensors are like multi-dimensional arrays with a uniform type
    """

    @staticmethod
    def open(audio_file_path):
        data, sampling_rate = torchaudio.load(audio_file_path)
        return (data, sampling_rate)

    """
    rechannel method: signals can either be mono or stereo. This method is used to get all our signals in the same dimensions.
    It converts all mono signals to stereo by duplicating the first channel
    Link for difference between mono/stereo : https://www.rowkin.com/blogs/rowkin/mono-vs-stereo-sound-whats-the-big-difference
    """

    # channels stereo/mono
    @staticmethod
    def rechannel(audio_file, new_channel):
        data, sampling_rate = audio_file

        if (data.shape[0] == new_channel):
            return audio_file

        if (new_channel == 1):
            # stereo to mono
            resig = data[:1, :]
        else:
            # mono to stereo by duplicating
            resig = torch.cat([data, data])

        return ((resig, sampling_rate))

    """
    resampling method: our audio signals have different sampling rates as well. Hence, We need to standardise the sampling rate.
    Different sampling rates result in different array sizes. Ex: sr - 40000Hz means array size of 400000 whereas 40010Hz means aaray size of 40010
    After standardisation we get all arrays of the same size
    """

    # resample one at a time and merge
    @staticmethod
    def resample(audio, new_sampling_rate):
        data, sampling_rate = audio

        if (sampling_rate == new_sampling_rate):
            return audio

        num_channels = data.shape[0]
        resig = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate)(data[:1, :])
        if (num_channels > 1):
            retwo = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate)(data[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, new_sampling_rate))

    """
    pad_trunc method: Our audio files are bound to be of different lengths of time. This also needs to be standardised.
    This method either extends the length by padding with silence (Zero Padding) or reduces the length by truncating
    """

    @staticmethod
    def pad_trunc(audio, max_ms):
        data, sampling_rate = audio
        num_rows, data_len = data.shape
        max_len = sampling_rate // 1000 * max_ms

        if (data_len > max_len):
            # truncate to given length
            data = data[:, :max_len]

        elif (data_len < max_len):
            # padding at the start and end of the audio
            pad_begin_len = random.randint(0,
                                           max_len - data_len)  # fill with random no between at 0 upto the extra time(maxlen-datalen)
            pad_end_len = max_len - data_len - pad_begin_len

            # Pad with 0s - Zero Padding
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            data = torch.cat((pad_begin, data, pad_end), 1)

        return (data, sampling_rate)

    # Spectrogram finally!!!
    """
    spectrogram method:
    Link for short explanation: https://colab.research.google.com/drive/1UgxygdrBfq7UGjhTCc9oupA-CyKFGhGa#scrollTo=733XclBe9Vgn
    """

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)


def data_processing(file):
    new_channel = 2  # making all stereo sounds
    new_sampling_rate = 44100  # permanently setting a standard rate
    duration = 6000  # setting a standard audio length of 6s, 6000ms

    # looping over every files in the folder of musical instrument for ex: sitar

    audio = AudioProcessing.open(file)
    resampled_audio = AudioProcessing.resample(audio, new_sampling_rate)
    rechanneled_audio = AudioProcessing.rechannel(resampled_audio, new_channel)
    padded_audio = AudioProcessing.pad_trunc(rechanneled_audio, duration)
    spectro_gram = AudioProcessing.spectro_gram(padded_audio, n_mels=64, n_fft=1024, hop_len=None)
    spec = np.array(spectro_gram)
    final_spec = spec.reshape((1,2*64*516))
    return final_spec



def load_model():
    model = keras.models.load_model('musical_instruments.h5')
    return model

class_names=["Violin", "MohanVeena", "Sitar"]

uploaded_file = st.file_uploader(label =" Upload a sound to test", type=['wav'] )
if uploaded_file is not None:
    st.audio(uploaded_file)
    model = load_model()
    processed_audio = data_processing(uploaded_file)
    prediction_prob = model.predict(processed_audio).reshape(-1)
    prediction_class = class_names[np.argmax(prediction_prob)]

    st.success(f"Audio belongs to class {prediction_class} with probability: {prediction_prob[np.argmax(prediction_prob)]}")



