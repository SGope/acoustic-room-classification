import os
import numpy as np
from torch.utils.data import Dataset
import torchaudio

class AudioDS(Dataset):
    def __init__(self, data, labels, path):
        """
        Function to initialise the dataset
        Args:
            param1: path: path to the input
        """
        self.data = data
        self.labels = labels
        self.data_path = path
        self.sr = 44100

    def __len__(self):
        """
        Function to find out the length of the dataset
        Returns:
            return1: no of files in the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Function to read the data files and obtain X (data) and Y (labels)
        Args:
            param1: item: name of the file
        Returns:
            return1: x: The data (melspectrogram or fft of the audio files depending on input of this training path)
            return2: y: The label for the data, in this thesis that is the name of the room
        """
        file_name = self.data[index]
        file_path = os.path.join(self.data_path, file_name)
        x = np.load(os.path.join(file_path), allow_pickle=True)
        label = self.labels[index]
        if label == "H1539b":
            y = 0
        elif label == "H1562":
            y = 1
        elif label == "H2505":
            y = 2
        elif label == "HL":
            y = 3
        elif label == "HU103":
            y = 4
        elif label == "LR001":
            y = 5
        elif label == "ML2-102":
            y = 6
        else:
            print("Wrong label")
            y = "invalid"
        return x, y

