import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def training(model, train_dl, val_dl, num_epochs):
    """ Training function for the neural network """
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    min_valid_loss = np.inf
    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []

    # Loss function, optimizer and scheduler
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=0.001, steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs, anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_val_loss = 0.0
        correct_prediction = 0
        correct_val_prediction = 0
        total_prediction = 0
        total_val_prediction = 0

        # Repeat for each batch in training set
        for inputs, labels in train_dl:
            # Get input features and labels and put them on the GPU
            #inputs, labels = data[0].to(device), data[1].to(device)
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Normalise the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs = model(inputs.unsqueeze(1))
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            #Print every 4 mini-batches
            # if i % 4 == 0:
            #     print('[%d, %5d] training loss: %.3f' % (epoch + 1, i + 1, running_loss / 4))

        with torch.set_grad_enabled(False):
            for inputs, labels in val_dl:
                # Get input features and labels and put them on the GPU
                #inputs, labels = data[0], data[1]
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Normalise the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # forward
                outputs = model(inputs.unsqueeze(1))
                val_loss = loss_func(outputs, labels)

                # Keep stats for Loss and Accuracy
                running_val_loss += val_loss.item()

                # Get the predicted class with highest score
                _, val_prediction = torch.max(outputs, 1)
                # Count of predictions that matched the target label
                correct_val_prediction += (val_prediction == labels).sum().item()
                total_val_prediction += val_prediction.shape[0]

                # Print every 4 mini-batches
                #if i % 4 == 0:
                #    print('[%d, %5d] validation loss: %.3f' % (epoch + 1, i + 1, running_val_loss / 4))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        val_batches = len(val_dl)
        avg_val_loss = running_val_loss / val_batches
        val_acc = correct_val_prediction / total_val_prediction
        train_acc.append(acc)
        valid_acc.append(val_acc)
        train_loss.append(avg_loss)
        valid_loss.append(avg_val_loss)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}, '
              f'Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {val_acc:.2f}')

        if min_valid_loss > avg_val_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{avg_val_loss:.6f}) \t Saving The Model')
            min_valid_loss = avg_val_loss
            # Saving State Dict
            torch.save(model.state_dict(), '/home/issac/PycharmProjects/room_classification/training_data/saved_model.pth')


    plt.figure(1, figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(valid_acc, label="val")
    plt.plot(train_acc, label="train")
    plt.ylim([0, 1.0])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("/home/issac/PycharmProjects/room_classification/training_data/accuracy.png")

    plt.figure(2, figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(valid_loss, label="val")
    plt.plot(train_loss, label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("/home/issac/PycharmProjects/room_classification/training_data/loss.png")
    #plt.show()

    print("Finished Training")