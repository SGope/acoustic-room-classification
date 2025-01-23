import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt


class Features():
    """To extract mel features for all convolved audio and fft features for all impulse recordings"""
    def extract_mel(self, file, branch):
        """To extract mel features from convolved audio"""
        print('Importing and processing:', file, '--please wait')
        audio, samplerate = librosa.load(os.path.join(branch, file), sr=44100)
        print(audio.shape)
        mel = librosa.feature.melspectrogram(audio, samplerate, n_mels=128, n_fft=2048, hop_length=512)
        mel_dB = librosa.power_to_db(mel, ref= np.max)
        mel_normalised = librosa.util.normalize(mel_dB)
        print(mel_normalised.shape)
        if mel_normalised.shape != (128, 147):
            print("!!!!!!!!!!!!!!!!!!!!!    NOT SAME SHAPE     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  #(431 for 5s, 147 for 1.7s)

        mel_path = "/home/issac/PycharmProjects/room_classification/mel/Final/Test/brir/"
        name = file.split('.')[0]
        np.save(os.path.join(mel_path, name), mel_normalised)

        # Saving mel plots as just image for input to LRP
        fig_path = "/home/issac/PycharmProjects/room_classification/LRP/mel/Final/Test/brir/"
        fig = plt.figure(2)
        img = librosa.display.specshow(mel_normalised, y_axis='mel', x_axis='time')
        # ax.set(title='Mel spectrogram display')
        fig.colorbar(img)
        fig.delaxes(fig.axes[1])
        plt.axis('off')
        plt.savefig(os.path.join(fig_path, name), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()
        #plt.show()

    def extract_fft(self, file, branch):
        """To extract fft features from impulse recordings"""
        print('Importing and processing:', file, '--please wait')
        audio, samplerate = librosa.load(os.path.join(branch, file), sr=44100)
        print(audio.shape)
        X = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=128)
        Xdb = librosa.amplitude_to_db(abs(X))
        X_normalised = librosa.util.normalize(Xdb)
        print(X_normalised.shape)
        if X_normalised.shape != (1025, 147):
            print("!!!!!!!!!!!!!!!!!!!!!    NOT SAME SHAPE     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        fft_path = "/home/issac/PycharmProjects/room_classification/mel/Final/Test/noise/"
        name = file.split('.')[0]
        np.save(os.path.join(fft_path, name), X_normalised)

        # Saving mel plots as just image for input to LRP
        fig_path = "/home/issac/PycharmProjects/room_classification/LRP/mel/Final/Test/noise/"
        fig = plt.figure(2)
        img = librosa.display.specshow(X_normalised, y_axis='fft', x_axis='time')
        # ax.set(title='Mel spectrogram display')
        fig.colorbar(img)
        fig.delaxes(fig.axes[1])
        plt.axis('off')
        plt.savefig(os.path.join(fig_path, name), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.clf()


    def get_files(self, folder, is_convolved):
        """To open the files and send for operations"""
        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/Final/Test/convolved/brir/"
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

        if (is_convolved == 1):
            for file in sorted(filenames):
                self.extract_mel(file, branch)
        elif (is_convolved == 2):
            for file in sorted(filenames):
                self.extract_fft(file, branch)
        print("Padding is completed")

def main():
    x = Features()
    folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    while (folder != 1) and (folder != 2) and (folder != 3):
        if (folder == 1) or (folder == 2) or (folder == 3):
            break
        folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    is_convolved = int(input("Enter 1 for convolved audio, 2 for impulses: "))
    while (is_convolved != 1) and (is_convolved != 2):
        if (is_convolved == 1) or (is_convolved == 2):
            break
        is_convolved = int(input("Enter 1 for convolved audio, 2 for impulses: "))
    x.get_files(folder, is_convolved)

if __name__ == "__main__":
    main()