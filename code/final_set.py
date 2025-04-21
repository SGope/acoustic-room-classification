import os
import random
from sklearn.model_selection import train_test_split
import shutil

def main():
    # Split set into test, train folders
    path = "/___ Enter path ___/BRIRs/DH/"
    train_path = "/___ Enter path ___/Final/Train/DH/"
    test_path = "/___ Enter path ___/Final/Test/DH/"
    filenames = [file for file in os.listdir(path) if file.endswith(".wav")]
    print(len(filenames))
    #random.shuffle(filenames)
    train_set, test_set = train_test_split(filenames, test_size=0.3, train_size=0.7, shuffle=True)
    for files in sorted(train_set):
        shutil.copy(os.path.join(path, files), os.path.join(train_path, files))
    for files in sorted(test_set):
        shutil.copy(os.path.join(path, files), os.path.join(test_path, files))
    print(len(train_set))
    print(len(test_set))

    precision = 10.987
    recall = 7.9876
    f1 = 8.9068
    lines = ['Readme: ', f'Precision: {precision: .2f}, Recall: {recall: .2f}, and F1 score: {f1: .2f}']
    with open('readme.txt', 'w') as f:
        f.writelines(lines)



if __name__ == "__main__":
    main()
