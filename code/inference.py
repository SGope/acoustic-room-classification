import os
import torch
from torch.utils.data import DataLoader
from data_preparation import AudioDS
from model import AudioClassifier
from torchsummary import summary
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

def inference(model, test_dl):
    """
    Function to get inference from model training, test set is used as input and we classify.
    """
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    correct_prediction = 0
    total_prediction = 0
    y_pred = []
    y_true = []

    # Disable gradient updates
    with torch.no_grad():
        for inputs, labels in test_dl:
            # Get the input features and labels
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Normalise the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs.unsqueeze(1))

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

    # Get precision, recall and f1 values
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, and F1 score: {f1:.2f}')
    lines = ['brir: ', f'\nAccuracy: {acc:.2f}, Total items: {total_prediction}', f'\nPrecision: {precision: .2f}, Recall: {recall: .2f}, and F1 score: {f1: .2f}']
    with open('/home/issac/PycharmProjects/room_classification/training_data/precision.txt', 'w') as f:
        f.writelines(lines)

    label = ['H1539b', 'H1562', 'H2505', 'HL', 'HU103', 'LR001', 'ML2-102']

    # Build confusion matrix
    plt.figure(1, figsize=(10, 7))
    cf_matrix = confusion_matrix(y_true, y_pred)
    ax = sn.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

    ax.set_title('Confusion Matrix with test results')
    plt.xlabel('Predicted Labels\n NOTE: All rooms have the same no. of inputs except LR001 which has 3 times less inputs')
    plt.ylabel('Actual Labels')

    ## Ticket labels - List must be in alphabetical order
    ax.set_xticklabels(label)
    ax.set_yticklabels(label)

    ## Display the visualization of the Confusion Matrix.
    plt.savefig("/home/issac/PycharmProjects/room_classification/training_data/cm_percent.png")
    plt.clf()

    #Confusion matrix in numbers
    plt.figure(2, figsize=(10, 7))
    ax = sn.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')

    ax.set_title('Confusion Matrix with training results')
    plt.xlabel('Predicted Labels\n NOTE: All rooms have the same no. of inputs except LR001 which has 3 times less inputs')
    plt.ylabel('Actual Labels')

    ## Ticket labels - List must be in alphabetical order
    ax.set_xticklabels(label)
    ax.set_yticklabels(label)

    ## Display the visualization of the Confusion Matrix.
    plt.savefig("/home/issac/PycharmProjects/room_classification/training_data/cm_numbers.png")


def main():
    """
    Run the inference on test dataset
    """
    # CUDA for PyTorch, create model and put it on CUDA
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    myModel = AudioClassifier()
    myModel = myModel.to(device)
    model_path = "/home/issac/PycharmProjects/room_classification/training_data/saved_model.pth"
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    print(myModel)
    summary(myModel, (1, 128, 147))

    # Parameters
    params = {'batch_size': 8,
              'shuffle': True}
    max_epochs = 50

    path = "/home/issac/PycharmProjects/room_classification/mel/Final/Test/brir/"
    print("----   MAKING LIST  ----")
    feature_dict = dict()
    label_dict = dict()
    filenames = [file for file in os.listdir(path) if file.endswith(".npy")]
    for index, file in enumerate(sorted(filenames)):
        feature_dict[index] = file
        label = file.split('_')[2]
        label_dict[index] = label
    print("----   LIST COMPLETED  ----")
    test_ds = AudioDS(feature_dict, label_dict, path)

    # Creating training and validation dataloaders
    print("----   CREATING DATALOADERS  ----")
    test_dl = DataLoader(test_ds, **params)

    # Run inference on trained model with the validation set
    inference(myModel, test_dl)


if __name__ == "__main__":
    main()