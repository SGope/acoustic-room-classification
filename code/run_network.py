import os
import torch
from torch.utils.data import random_split, DataLoader
from data_preparation import AudioDS
from model import AudioClassifier
from training import training
from torchsummary import summary
import time as tm

def main():
    """To prepare the training and validation datasets and perform training"""
    # CUDA for PyTorch, create model and put it on CUDA
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(device)
    torch.backends.cudnn.benchmark = True
    myModel = AudioClassifier()
    myModel = myModel.to(device)
    print(myModel)
    summary(myModel, (1, 128, 147))

    # Parameters
    params = {'batch_size': 8,
              'shuffle': True}
    max_epochs = 50

    path = "/___ Enter path ___/Train/brir/"
    print("----   MAKING LIST  ----")
    feature_dict = dict()
    label_dict = dict()
    filenames = [file for file in os.listdir(path) if file.endswith(".npy")]
    for index, file in enumerate(sorted(filenames)):
        feature_dict[index] = file
        label = file.split('_')[2]
        label_dict[index] = label
    print("----   LIST COMPLETED  ----")
    audio_ds = AudioDS(feature_dict, label_dict, path)

    #Random split data into training and validation set in ratio of 7:3
    num_items = len(audio_ds)
    train_num = round(num_items * 0.7)
    val_num = num_items - train_num
    train_ds, val_ds = random_split(audio_ds, [train_num, val_num])

    #Creating training and validation dataloaders
    print("----   CREATING DATALOADERS  ----")
    train_dl = DataLoader(train_ds, **params)
    val_dl = DataLoader(val_ds, **params)

    print("----   TRAINING MODEL  ----")
    start_time = tm.time()
    training(myModel, train_dl, val_dl, max_epochs)
    print("--- %s Time taken in seconds for training ---" % (tm.time() - start_time))


if __name__ == "__main__":
    main()
