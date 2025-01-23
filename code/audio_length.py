import numpy as np
import pickle
import os
import librosa

class Length():
    def read_length(self, file, branch):
        """
        To find the length of all audios and save them in a file.
        """
        length = []
        audio, samplerate = librosa.load(os.path.join(branch, file), sr=None)
        duration = len(audio) / samplerate
        return duration

    def open_files(self, folder):
        """
        Get the file and perform required actions
        """
        length = []
        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/trial/"
        #branch = "/home/issac/PycharmProjects/room_classification/dataset_for_students/trial/audio_for_convolution/"
        #filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]

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
            #duration = self.find_length(file, branch)
            audio, samplerate = librosa.load(os.path.join(branch, file), sr=44100)
            duration = len(audio) / samplerate
            length.append(duration)

        length_array = np.array(length)
        with open('length.txt', mode='wb') as f:
            pickle.dump(length_array, f)
        print("Plots have been saved")
        print(length_array)



def main():
    x = Length()
    folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    while (folder != 1) and (folder != 2) and (folder != 3):
        if (folder == 1) or (folder == 2) or (folder == 3):
            break
        folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    x.open_files(folder)

if __name__ == "__main__":
    main()