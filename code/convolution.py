import os
import librosa
import numpy as np
from wave import open
import soundfile


class Wave:
    def __init__(self, data, frame_rate):
        self.data = normalize(data)
        self.frame_rate = frame_rate

    def make_spectrum(self):
        amplitudes = np.fft.rfft(self.data)
        frequencies = np.fft.rfftfreq(len(self.data), 1 / self.frame_rate)

        return Spectrum(amplitudes, frequencies, self.frame_rate)

    def zero_padding(self, n):
        zeros = np.zeros(n)
        zeros[:len(self.data)] = self.data

        self.data = zeros

    def cut(self, n):
        zeros = np.zeros(n)
        zeros = self.data[:n]
        self.data = zeros

    def write(self, file):
        reader = open(file, 'w')

        reader.setnchannels(1)
        reader.setsampwidth(2)
        reader.setframerate(self.frame_rate)

        frames = self.quantize().tostring()
        reader.writeframes(frames)

        reader.close()

    def quantize(self):
        if max(self.data) > 1 or min(self.data) < -1:
            self.data = normalize(self.data)

        return (self.data * 32767).astype(np.int16)


class Spectrum:
    def __init__(self, amplitudes, frequencies, frame_rate):
        self.amplitudes = np.asanyarray(amplitudes)
        self.frequencies = np.asanyarray(frequencies)
        self.frame_rate = frame_rate

    def __mul__(self, other):
        return Spectrum(self.amplitudes * other.amplitudes, self.frequencies, self.frame_rate)

    def make_wave(self):
        return Wave(np.fft.irfft(self.amplitudes), self.frame_rate)


def convert_wav(file):
    data, samprate = soundfile.read(file)
    soundfile.write(file, data, samprate, subtype='PCM_16')


def read_wave(file):
    reader = open(file)

    _, sampwidth, framerate, nframes, _, _ = reader.getparams()
    frames = reader.readframes(nframes)

    reader.close()

    dtypes = {1: np.int8, 2: np.int16, 4: np.int32}

    if sampwidth not in dtypes:
        raise ValueError('unsupported sample width')

    data = np.frombuffer(frames, dtype=dtypes[sampwidth])

    num_channels = reader.getnchannels()
    if num_channels == 2:
        data = data[::2]

    return Wave(data, framerate)


def normalize(data):
    high, low = abs(max(data)), abs(min(data))
    return data / max(high, low)


def convolution_reverb(audio_file, ir_file, output_file):
    convert_wav(audio_file)
    convert_wav(ir_file)

    audio = read_wave(audio_file)
    ir = read_wave(ir_file)

    if len(audio.data) > len(ir.data):
        ir.zero_padding(len(audio.data))
        #audio.cut(len(ir.data))
        print("Audio is longer")
        print(len(audio.data))
        print(len(ir.data))
    else:
        audio.zero_padding(len(ir.data))
        print("IR is longer")
        print(len(audio.data))
        print(len(ir.data))

    ir_spectrum = ir.make_spectrum()
    audio_spectrum = audio.make_spectrum()

    convolution = audio_spectrum * ir_spectrum
    wave = convolution.make_wave()
    wave.write(output_file)




class Convole():
    def convolve_audio(self, file, branch, folder):
        """Convolve the audio clip with the impulse"""
        audio_path = "/___ Enter path ___/audio_for_convolution/uncut/"
        print(file)
        ir_wave = os.path.join(branch, file)
        audio_files = [filename for filename in os.listdir(audio_path) if filename.endswith(".wav")]
        output_path = "/___ Enter path ___/Test/convolved/brir_uncut/"
        if folder == 1:
            branch = os.path.join(output_path, "DH/")
        elif folder == 2:
            branch = os.path.join(output_path, "MTB/")
        else:
            branch = os.path.join(output_path, "SDM_KEMAR/")

        for filename in sorted(audio_files):
            audio_wav = os.path.join(audio_path, filename)
            name = file.split('.')[0]
            name = name + '_' + filename
            output_wave = os.path.join(branch, name)
            try:
                convolution_reverb(audio_wav, ir_wave, output_wave)
            except:
                print("Not done for: ", filename)
        print("Done for: ", file)

    def get_files(self, folder):
        """To get files for processing"""
        path = "/___ Enter path ___/Test/"
        if folder == 1:
            branch = os.path.join(path, "DH/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        elif folder == 2:
            branch = os.path.join(path, "MTB/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        else:
            branch = os.path.join(path, "SDM_KEMAR/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        print("Current folder is: ", branch)

        for file in sorted(filenames):
            self.convolve_audio(file, branch, folder)
        print("Plots have been saved")

def main():
    x = Convole()
    folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    while (folder != 1) and (folder != 2) and (folder != 3):
        if (folder == 1) or (folder == 2) or (folder == 3):
            break
        folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    x.get_files(folder)

if __name__ == "__main__":
    main()
