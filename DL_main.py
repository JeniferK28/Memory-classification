
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import  pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import time
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from ptl_cm import plot_confusion_matrix
from Train_pred import trainet,prediction
from Shallow_Net import  Net
from loss_plt import loss_plt



def DL_main (X_train, X_test, y_train, y_test,args):
    batch_size=args.batch_size
    np.random.seed(seed=args.seed)
    train_size = np.shape(X_train)
    test_size = np.shape(X_test)

    #X_train = X_train.reshape(train_size[0], 1, train_size[1], train_size[2])
    #X_test = X_test.reshape(test_size[0], 1, test_size[1], test_size[2])

    y_train = Variable(Tensor(y_train)).type(torch.LongTensor)
    y_test = Variable(Tensor(y_test)).type(torch.LongTensor)

    # Constuye train Data loader
    train_data = torch.utils.data.TensorDataset(Tensor(X_train), y_train)
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Construye test Data Loader
    test_data = torch.utils.data.TensorDataset(Tensor(X_test), y_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True)

    #GPU uso
    use_cuda = torch.cuda.is_available()

    #Llama Net
    #net = Net(input_dim, hidden_dim, layer_dim, output_dim).cuda(0)
    net=Net().cuda(0)
    # loss
    criterion = nn.CrossEntropyLoss()
    # back propagation
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    # hyper_parameter
    num_epochs = args.num_epochs
    net, train_loss=trainet(data_loader, net, num_epochs, criterion, optimizer)

    pred_acc=prediction(test_loader,net)

    return pred_acc
