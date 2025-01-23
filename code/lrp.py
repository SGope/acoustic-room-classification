import copy
import torch.nn
import numpy as np
import os
from data_preparation import AudioDS
from model import AudioClassifier
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import make_axes_locatable

def LRP_individual(model, X, device):
    """Function to back propagate through the model and apply LRP"""
    # Getting list of layers
    layers = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)][1:]

    # Propagate input through the network
    L = len(layers)
    A = [X] + [X] * L       # List to store activation in each layer
    for layer in range(L):
        if layer == 13:
            A[layer] = torch.reshape(A[layer], (A[layer].shape[0], A[layer].shape[1]))
        A[layer + 1] = layers[layer].forward(A[layer])

    # Get relevance of the last layer using highest classification score of the top layer
    T = A[-1].cpu().detach().numpy().tolist()[0]
    index = T.index(max(T))
    T = np.abs(np.array(T)) * 0
    T[index] = 1
    T = torch.FloatTensor(T)
    # Make list of relevances with (L+1) layers and assign relevance of last one
    R = [None] * L + [(A[-1].cpu() * T).data + 1e-6]

    # Propagation from top layer to following layers
    for layer in range(0, L)[::-1]:

        if isinstance(layers[layer], torch.nn.Conv2d) \
                or isinstance(layers[layer], torch.nn.BatchNorm2d) \
                or isinstance(layers[layer], torch.nn.AdaptiveAvgPool2d) or isinstance(layers[layer], torch.nn.Linear):

            # Rho function to be applied
            if 0 < layer <= 8:            # Gamma rule (LRP-gamma)
                rho = lambda p: p + 0.25 * p.clamp(min=0); incr = lambda z: z+1e-9
            else:                         # Basic rule (LRP-0)
                rho = lambda p: p; incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data

            # print("process started", layer)
            #print(layers[layer].weight)
            A[layer] = A[layer].data.requires_grad_(True)
            # Step 1: Transform the weights of the layer and executes a forward pass
            if isinstance(layers[layer], torch.nn.AdaptiveAvgPool2d):
                flag = 1
            else:
                flag = 0
            z = newlayer(layers[layer], rho, flag).forward(A[layer]) + 1e-9
            # Step 2: Element-wise division between the relevance of the next layer and the denominator
            # print(R[layer + 1].shape)
            # print(z.shape)
            s = (R[layer + 1].to(device) / z).data
            # Step 3: Calculate the gradient and multiply it by the activation layer
            (z * s).sum().backward()
            c = A[layer].grad
            R[layer] = (A[layer] * c).cpu().data

            if layer == 13:
                R[layer] = torch.reshape(R[layer], (R[layer].shape[0], R[layer].shape[1], 1, 1))

        else:
            R[layer] = R[layer + 1]

    # print("Layer 13 sum", np.sum(R[13].data.numpy()))
    # print("Layer 12 sum", np.sum(R[12].data.numpy()))
    # print("Layer 11 sum", np.sum(R[11].data.numpy()))
    # print("Layer 10 sum", np.sum(R[10].data.numpy()))
    # print("Layer 9 sum", np.sum(R[9].data.numpy()))
    # print("Layer 8 sum", np.sum(R[8].data.numpy()))
    # print("Layer 7 sum", np.sum(R[7].data.numpy()))
    # print("Layer 6 sum", np.sum(R[6].data.numpy()))
    # print("Layer 5 sum", np.sum(R[5].data.numpy()))
    # print("Layer 4 sum", np.sum(R[4].data.numpy()))
    # print("Layer 3 sum", np.sum(R[3].data.numpy()))
    # print("Layer 2 sum", np.sum(R[2].data.numpy()))
    # print("Layer 1 sum", np.sum(R[1].data.numpy()))
    # Return relevance of input layer
    return R[0]

def newlayer(layer, g, flag):
    """Clone a layer and pass its parameters through a function g"""
    layer_copy = copy.deepcopy(layer)
    if flag == 1:
        return layer_copy
    elif flag == 0:
        layer_copy.weight = torch.nn.Parameter(g(layer.weight))
        layer_copy.bias = torch.nn.Parameter(g(layer.bias))
        return layer_copy




def lrp(filename):
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device("cpu")
    torch.backends.cudnn.benchmark = True

    # Create model and load saved model in evaluation mode
    myModel = AudioClassifier()
    myModel = myModel.to(device)
    model_path = "/home/issac/PycharmProjects/room_classification/training_data/saved_model.pth"
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()

    """
    file_path = "/home/issac/PycharmProjects/room_classification/mel/smaller_path_2/1.7s/BRIR_DH_H1539b_E_LS_000_Claps.npy"
    item = np.load(file_path, allow_pickle=True)
    """

    path = "/home/issac/PycharmProjects/room_classification/mel/Final/Train/noise/"
    # print("----   MAKING LIST  ----")
    feature_dict = dict()
    label_dict = dict()
    filenames = [file for file in os.listdir(path) if file.endswith(filename)]
    for index, file in enumerate(sorted(filenames)):
        print("Current file", file)
        feature_dict[index] = file
        label = file.split('_')[2]
        label_dict[index] = label
    # print("----   LIST COMPLETED  ----")
    audio_ds = AudioDS(feature_dict, label_dict, path)
    audio_dl = DataLoader(audio_ds, shuffle=False)
    for inputs, labels in audio_dl:
        item = inputs.unsqueeze(1)

    Rel = LRP_individual(myModel, item, device="cpu")

    # Plot the relevance for each channel
    minv = np.min(np.min(np.min(Rel.data.numpy(), axis=2), axis=2), axis=1)  # Get the minimum relevance
    maxv = np.max(np.max(np.max(Rel.data.numpy(), axis=2), axis=2), axis=1)  # Get the maximum relevance

    # fig, axs = plt.subplots(3, 6)
    # count = 0
    # for i in range(3):
    #     for j in range(6):
    #         im = axs[i, j].imshow(Rel.data.numpy()[0, count, :, :], vmin=minv, vmax=maxv)
    #         count += 1
    # fig.subplots_adjust(right=0.8)
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(Rel.data.numpy()[0, 0, :, :], vmin=minv, vmax=maxv, cmap='gray')
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    fig_path = "/home/issac/PycharmProjects/room_classification/LRP/relevance/Final/Train/noise/"
    name = filename.split('.')[0]
    fig.delaxes(fig.axes[1])
    plt.axis('off')
    plt.savefig(os.path.join(fig_path, name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()
    #plt.savefig("/home/issac/High resoltion.png", dpi=300)
    #plt.show()

def main():
    path = "/home/issac/PycharmProjects/room_classification/mel/Final/Train/noise/"
    filenames = [file for file in os.listdir(path) if file.endswith(".npy")]
    for index, file in enumerate(sorted(filenames)):
        lrp(file)
        print(file, " has been processed")

if __name__ == "__main__":
    main()
