import os
import librosa
import matplotlib.pyplot as plt
import librosa.display


class Spectrogram():
    def plot(self, file, branch, folder):
        """To make the plots"""
        print(file)
        audio, samplerate = librosa.load(os.path.join(branch, file), sr=None)
        X = librosa.stft(audio)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(1)
        librosa.display.specshow(Xdb, sr = samplerate, x_axis = 'time', y_axis = 'hz')
        plt.colorbar()
        plt.xlabel("Time (in seconds) -->")
        plt.ylabel("Frequency (Hz) in linear scale")
        plt.title(file)
        save_location = "/home/issac/PycharmProjects/room_classification/dataset_for_students/trial/noise_2/spectrogram/"
        name = file.split('.')[0]
        if folder == 1:
            path = os.path.join(save_location, "DH/", name)
        elif folder == 2:
            path = os.path.join(save_location, "MTB/", name)
        else:
            path = os.path.join(save_location, "SDM_KEMAR/", name)
        plt.savefig(path + '.png')
        #plt.show()
        plt.clf()

    def get_files(self, folder):
        """To get files for processing"""
        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/trial/noise_2/"
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
            self.plot(file, branch, folder)
        print("Plots have been saved")

def main():
    x = Spectrogram()
    folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    while (folder != 1) and (folder != 2) and (folder != 3):
        if (folder == 1) or (folder == 2) or (folder == 3):
            break
        folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    x.get_files(folder)

if __name__ == "__main__":
    main()