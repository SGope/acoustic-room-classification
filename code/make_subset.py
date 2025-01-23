import os
import fnmatch

class Subset():
    """
    To make subset of the dataset for training, we take middle configuration in each room so all rooms are depicted.
    """

    def get_files(self, folder):
        count = 0
        path = "/home/issac/PycharmProjects/room_classification/dataset_for_students/smaller_path_2/"
        if folder == 1:
            branch = os.path.join(path, "DH/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        elif folder == 2:
            branch = os.path.join(path, "MTB/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        else:
            branch = os.path.join(path, "SDM_KEMAR/")
            filenames = [file for file in os.listdir(branch) if file.endswith(".wav")]
        """
        #Already did this part
        for file in filenames:
            if fnmatch.fnmatch(file, "*_RS_*.wav"):
                os.remove(os.path.join(branch, file))
                count += 1
            #else:
        """


        for file in sorted(filenames):
             number = int(file[-7:-4])
             count += 1
             if (number % 240) != 0:
                 os.remove(os.path.join(branch, file))
                 count -= 1

        print(count)

def main():
    x = Subset()
    folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    while (folder != 1) and (folder != 2) and (folder != 3):
        if (folder == 1) or (folder == 2) or (folder == 3):
            break
        folder = int(input("Enter 1 for DH, 2 for MTB and 3 for SDM_KEMAR: "))
    x.get_files(folder)

if __name__ == "__main__":
    main()