import math
import numpy as np
import os
import librosa
from librosa import display
import matplotlib.pyplot as plt
from scipy.fftpack import fft


class Plots():
    """
    For saving plots of audio signal and fft spectrum
    """

    def make_plots(self, file, branch, folder):
        print(file)
        audio, samplerate = librosa.load(os.path.join(branch, file), sr=None)
        plt.figure(1)
        librosa.display.waveplot(y = audio, sr = samplerate)
        plt.xlabel("Time (in seconds) -->")
        plt.ylabel("Amplitude")
        save_location = "___ Enter path ___"
        name = file.split('.')[0]
        print(name)
        if folder == 1:
            path = os.path.join(save_location, "DH/", name)
        elif folder == 2:
            path = os.path.join(save_location, "MTB/", name)
        else:
            path = os.path.join(save_location, "SDM_KEMAR/", name)
        plt.savefig(path + '.png')
        plt.clf()

        n = len(audio)
        T = 1/samplerate
        yf = fft(audio)
        xf = np.linspace(0.0, 1.0/(2.0*T), math.floor(n/2))
        plt.figure(2)
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
        plt.grid()
        plt.xlabel("Frequency -->")
        plt.ylabel("Magnitude")
        save_location = "___ Enter path ___"
        if folder == 1:
            path = os.path.join(save_location, "DH/", name)
        elif folder == 2:
            path = os.path.join(save_location, "MTB/", name)
        else:
            path = os.path.join(save_location, "SDM_KEMAR/", name)
        plt.savefig(path + '.png')

        print("done")

    def get_files(self, folder):
        path = "___ Enter path ___"
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
            self.make_plots(file, branch, folder)
        print("Plots have been saved")

def main():
    x = Plots()
    folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    while (folder != 1) and (folder != 2) and (folder != 3):
        if (folder == 1) or (folder == 2) or (folder == 3):
            break
        folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    x.get_files(folder)

if __name__ == "__main__":
    main()
